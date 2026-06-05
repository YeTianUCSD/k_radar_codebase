from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchhd import embeddings
except ImportError as exc:
    embeddings = None
    _TORCHHD_IMPORT_ERROR = exc
else:
    _TORCHHD_IMPORT_ERROR = None


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _hard_quantize(x: torch.Tensor) -> torch.Tensor:
    out = x.sign()
    out.masked_fill_(out == 0, 1)
    return out


def _ste_quantize(x: torch.Tensor) -> torch.Tensor:
    hard = _hard_quantize(x)
    return x + (hard - x).detach()


@dataclass
class HDConfig:
    feat_dim: int
    num_classes: int
    num_anchors: int
    use_background: bool = False
    hd_dim: int = 10000
    encoder: str = 'rp'
    quantize: bool = True
    temperature: float = 1.0
    logit_scale: float = 1.0
    seed: int = 0
    encode_chunk: int = 8192
    num_levels: int = 100
    randomness: float = 0.0
    train_quantize: Optional[Union[bool, str]] = None


class HDEmbedder(nn.Module):
    def __init__(self, cfg: HDConfig):
        super().__init__()
        if embeddings is None:
            raise ImportError(
                "torchhd is required for HDCore. Install it in the active environment, "
                "for example: pip install torchhd"
            ) from _TORCHHD_IMPORT_ERROR

        self.cfg = cfg
        torch.manual_seed(int(cfg.seed))
        enc = str(cfg.encoder).lower()
        if enc == 'rp':
            self.encoder = embeddings.Projection(int(cfg.feat_dim), int(cfg.hd_dim))
        elif enc == 'level':
            self.value = embeddings.Level(int(cfg.num_levels), int(cfg.hd_dim), randomness=float(cfg.randomness))
            self.position = embeddings.Random(int(cfg.feat_dim), int(cfg.hd_dim))
            self.encoder = None
        elif enc == 'sinusoid':
            self.encoder = embeddings.Sinusoid(int(cfg.feat_dim), int(cfg.hd_dim))
        else:
            raise ValueError(f'Unsupported HD encoder: {cfg.encoder}')

        for param in self.parameters():
            param.requires_grad_(False)

    def _encode_level(self, feat: torch.Tensor) -> torch.Tensor:
        n, c = feat.shape
        if c != int(self.cfg.feat_dim):
            raise RuntimeError(f'HD feature dim mismatch: got {c}, expected {self.cfg.feat_dim}')
        f_min = feat.min(dim=0, keepdim=True).values
        f_max = feat.max(dim=0, keepdim=True).values
        feat01 = (feat - f_min) / (f_max - f_min).clamp_min(1e-6)
        idx = torch.clamp((feat01 * (int(self.cfg.num_levels) - 1)).long(), 0, int(self.cfg.num_levels) - 1)
        value = self.value.weight.index_select(0, idx.reshape(-1)).view(n, c, int(self.cfg.hd_dim))
        position = self.position.weight.unsqueeze(0).expand(n, c, int(self.cfg.hd_dim))
        # Avoid importing torchhd.functional in the hot path; multiplication is binding for bipolar vectors.
        return (value * position).sum(dim=1)

    def forward(self, feat: torch.Tensor, quantize: Optional[Union[bool, str]] = None) -> torch.Tensor:
        if feat.dim() != 2:
            raise RuntimeError(f'Expected HD input [N, C], got {tuple(feat.shape)}')
        feat = feat.float()
        enc = str(self.cfg.encoder).lower()
        if enc == 'level':
            hv = self._encode_level(feat)
        else:
            hv = self.encoder(feat)
        quantize_mode = self.cfg.quantize if quantize is None else quantize
        if isinstance(quantize_mode, str):
            quantize_mode = quantize_mode.lower()
            if quantize_mode == 'ste':
                hv = _ste_quantize(hv)
            elif quantize_mode in ('true', '1', 'yes'):
                hv = _hard_quantize(hv)
            elif quantize_mode not in ('false', '0', 'no'):
                raise ValueError(f'Unsupported quantize mode: {quantize_mode}')
        elif bool(quantize_mode):
            hv = _hard_quantize(hv)
        return hv.float()

    def forward_chunked(self, feat: torch.Tensor, chunk: int = 8192, quantize: Optional[Union[bool, str]] = None) -> torch.Tensor:
        chunk = int(chunk)
        if chunk <= 0 or feat.shape[0] <= chunk:
            return self.forward(feat, quantize=quantize)
        outs = []
        for start in range(0, feat.shape[0], chunk):
            outs.append(self.forward(feat[start:start + chunk], quantize=quantize))
        return torch.cat(outs, dim=0)


class HDMemory(nn.Module):
    def __init__(self, num_classes: int, hd_dim: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hd_dim = int(hd_dim)
        self.register_buffer('classify_weights', torch.zeros(self.num_classes, self.hd_dim), persistent=True)
        self.register_buffer('prototypes', torch.zeros(self.num_classes, self.hd_dim), persistent=True)

    @torch.no_grad()
    def reset(self):
        self.classify_weights.zero_()
        self.prototypes.zero_()

    @torch.no_grad()
    def normalize_(self):
        self.prototypes.copy_(_normalize_rows(self.classify_weights))

    @torch.no_grad()
    def add_(self, labels: torch.Tensor, hv: torch.Tensor, alpha: float = 1.0):
        if labels.numel() == 0:
            return
        labels = labels.to(device=self.classify_weights.device, dtype=torch.long)
        hv = hv.to(device=self.classify_weights.device, dtype=self.classify_weights.dtype)
        if float(alpha) != 1.0:
            hv = hv * float(alpha)
        self.classify_weights.index_add_(0, labels, hv)

    def logits(self, hv: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        hv = hv.to(device=self.prototypes.device, dtype=self.prototypes.dtype)
        logits = _normalize_rows(hv) @ self.prototypes.t()
        if float(temperature) != 1.0:
            logits = logits / float(temperature)
        return logits


class HDCore(nn.Module):
    def __init__(self, cfg: HDConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer('logit_scale', torch.tensor(float(cfg.logit_scale), dtype=torch.float32), persistent=True)
        self.embedder = HDEmbedder(cfg)
        memory_classes = cfg.num_classes + 1 if cfg.use_background else cfg.num_classes
        self.memory = HDMemory(memory_classes, cfg.hd_dim)

    @staticmethod
    def from_head_cfg(hd_cfg: Any, feat_dim: int, num_classes: int, num_anchors: int) -> 'HDCore':
        enc_cfg = _cfg_get(hd_cfg, 'ENCODER_CFG', {})
        cfg = HDConfig(
            feat_dim=int(feat_dim) + int(num_anchors),
            num_classes=int(num_classes),
            num_anchors=int(num_anchors),
            use_background=bool(_cfg_get(hd_cfg, 'USE_BACKGROUND', False)),
            hd_dim=int(_cfg_get(hd_cfg, 'HD_DIM', 10000)),
            encoder=str(_cfg_get(hd_cfg, 'ENCODER', 'rp')).lower(),
            quantize=bool(_cfg_get(hd_cfg, 'QUANTIZE', True)),
            temperature=float(_cfg_get(hd_cfg, 'TEMPERATURE', 1.0)),
            logit_scale=float(_cfg_get(hd_cfg, 'LOGIT_SCALE', 1.0)),
            seed=int(_cfg_get(hd_cfg, 'SEED', 0)),
            encode_chunk=int(_cfg_get(hd_cfg, 'ENCODE_CHUNK', _cfg_get(enc_cfg, 'ENCODE_CHUNK', 8192))),
            num_levels=int(_cfg_get(enc_cfg, 'NUM_LEVELS', 100)),
            randomness=float(_cfg_get(enc_cfg, 'RANDOMNESS', 0.0)),
            train_quantize=_cfg_get(hd_cfg, 'TRAIN_QUANTIZE', None),
        )
        return HDCore(cfg)

    def make_anchor_features(self, feat_map: torch.Tensor) -> torch.Tensor:
        if feat_map.dim() != 4:
            raise RuntimeError(f'Expected feature map [B, C, H, W], got {tuple(feat_map.shape)}')
        b, c, h, w = feat_map.shape
        a = int(self.cfg.num_anchors)
        cell_feat = feat_map.permute(0, 2, 3, 1).contiguous()
        cell_feat = cell_feat.unsqueeze(3).expand(b, h, w, a, c)
        eye = torch.eye(a, device=feat_map.device, dtype=feat_map.dtype)
        anchor_one_hot = eye.view(1, 1, 1, a, a).expand(b, h, w, a, a)
        return torch.cat([cell_feat, anchor_one_hot], dim=-1).reshape(b * h * w * a, c + a)

    def logits_from_feature_map(self, feat_map: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat_map.shape
        a = int(self.cfg.num_anchors)
        k = int(self.cfg.num_classes)
        feat = self.make_anchor_features(feat_map)
        chunk = int(self.cfg.encode_chunk)
        logits_out = []
        for start in range(0, feat.shape[0], chunk if chunk > 0 else feat.shape[0]):
            feat_chunk = feat[start:start + (chunk if chunk > 0 else feat.shape[0])]
            quantize = self.cfg.quantize
            if self.training and self.cfg.train_quantize is not None:
                quantize = self.cfg.train_quantize
            hv = self.embedder(feat_chunk, quantize=quantize)
            logits_out.append(self.memory.logits(hv, temperature=float(self.cfg.temperature)))
        logits = torch.cat(logits_out, dim=0)
        if bool(self.cfg.use_background):
            bg_logits = logits[:, :1]
            logits = logits[:, 1:] - bg_logits
        if float(self.logit_scale.item()) != 1.0:
            logits = logits * self.logit_scale.to(device=logits.device, dtype=logits.dtype)
        return logits.view(b, h, w, a, k).reshape(b, h, w, a * k).contiguous()

    def _labels_to_memory_indices(self, labels_1based: torch.Tensor) -> torch.Tensor:
        labels = labels_1based.long()
        if bool(self.cfg.use_background):
            return labels.clamp(min=0, max=int(self.cfg.num_classes))
        if labels.numel() > 0 and int(labels.min().item()) >= 1:
            labels = labels - 1
        return labels

    @torch.no_grad()
    def build_update(self, feat_anchor: torch.Tensor, labels_1based: torch.Tensor, alpha: float = 1.0):
        if feat_anchor.numel() == 0:
            return 0
        labels = self._labels_to_memory_indices(labels_1based)
        hv = self.embedder.forward_chunked(feat_anchor, chunk=int(self.cfg.encode_chunk))
        hv = F.normalize(hv.float(), p=2, dim=1)
        self.memory.add_(labels, hv, alpha=float(alpha))
        return int(labels.numel())

    @torch.no_grad()
    def adaptive_update(self, feat_anchor: torch.Tensor, labels_1based: torch.Tensor, alpha: float = 1.0) -> Dict[str, int]:
        if feat_anchor.numel() == 0:
            return {'num_total': 0, 'num_bg': 0, 'num_pos': 0, 'num_correct': 0, 'num_wrong': 0}
        labels = self._labels_to_memory_indices(labels_1based)
        hv = self.embedder.forward_chunked(feat_anchor, chunk=int(self.cfg.encode_chunk))
        hv = F.normalize(hv.float(), p=2, dim=1)
        logits = self.memory.logits(hv, temperature=float(self.cfg.temperature))
        pred = logits.argmax(dim=1).long()
        wrong = pred != labels

        self.memory.add_(labels, hv, alpha=alpha)
        if wrong.any():
            self.memory.add_(pred[wrong], -hv[wrong], alpha=alpha)
        self.memory.normalize_()
        num_bg = int((labels == 0).sum().item()) if bool(self.cfg.use_background) else 0
        num_pos = int((labels > 0).sum().item()) if bool(self.cfg.use_background) else int(labels.numel())
        return {
            'num_total': int(labels.numel()),
            'num_bg': num_bg,
            'num_pos': num_pos,
            'num_correct': int((~wrong).sum().item()),
            'num_wrong': int(wrong.sum().item()),
        }

    @torch.no_grad()
    def save_memory(self, path: str, meta: Optional[Dict[str, Any]] = None):
        torch.save({
            'cfg': self.cfg.__dict__.copy(),
            'embedder': self.embedder.state_dict(),
            'memory': self.memory.state_dict(),
            'meta': meta or {},
        }, path)

    @torch.no_grad()
    def load_memory(self, path: str, strict: bool = True, map_location: str = 'cpu'):
        payload = torch.load(path, map_location=map_location)
        if 'embedder' in payload:
            self.embedder.load_state_dict(payload['embedder'], strict=strict)
        if 'memory' in payload:
            memory_state = payload['memory']
            can_expand_fg_memory = (
                bool(self.cfg.use_background)
                and memory_state.get('classify_weights', None) is not None
                and memory_state['classify_weights'].shape[0] == int(self.cfg.num_classes)
                and self.memory.classify_weights.shape[0] == int(self.cfg.num_classes) + 1
            )
            if can_expand_fg_memory:
                self.memory.reset()
                self.memory.classify_weights[1:].copy_(memory_state['classify_weights'].to(self.memory.classify_weights.device))
                self.memory.prototypes[1:].copy_(memory_state['prototypes'].to(self.memory.prototypes.device))
            else:
                self.memory.load_state_dict(memory_state, strict=strict)
        elif 'classify_weights' in payload and 'prototypes' in payload:
            self.memory.classify_weights.copy_(payload['classify_weights'].to(self.memory.classify_weights.device))
            self.memory.prototypes.copy_(payload['prototypes'].to(self.memory.prototypes.device))
        self.memory.normalize_()
        return payload.get('meta', {})
