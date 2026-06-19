import torch
import torch.nn as nn
import torch.nn.functional as F

from .scene_context import SceneContextRegistry


def _validate_scene_name(scene_name):
    if scene_name is None:
        raise RuntimeError('scene_context is required for PSP layers')
    return str(scene_name)


def _broadcast_last_dim(context, x):
    return context.view(*([1] * (x.dim() - 1)), -1)


class PSPContextModule(nn.Module):
    def __init__(self, key_name, seed=20260619, enabled=True):
        super().__init__()
        self.key_name = str(key_name)
        self.scene_context_registry = SceneContextRegistry(seed=seed, enabled=enabled)
        self.active_scene_context = None

    def set_scene_context(self, scene_name):
        self.active_scene_context = None if scene_name is None else str(scene_name)


class PSPLayerNorm(PSPContextModule):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, key_name='psp.ln', seed=20260619, enabled=True):
        super().__init__(key_name=key_name, seed=seed, enabled=enabled)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(int(x) for x in normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    @classmethod
    def from_layer_norm(cls, layer_norm, key_name, seed=20260619, enabled=True):
        module = cls(
            normalized_shape=layer_norm.normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.elementwise_affine,
            key_name=key_name,
            seed=seed,
            enabled=enabled,
        )
        if layer_norm.elementwise_affine:
            module.weight.data.copy_(layer_norm.weight.data)
            module.bias.data.copy_(layer_norm.bias.data)
        return module

    def forward(self, x):
        weight = self.weight
        bias = self.bias
        if self.elementwise_affine and self.scene_context_registry.enabled:
            scene_name = _validate_scene_name(self.active_scene_context)
            weight_context = self.scene_context_registry.get_tensor(
                scene_name=scene_name,
                key_name=f'{self.key_name}.weight',
                shape=self.normalized_shape,
                device=x.device,
                dtype=x.dtype,
            )
            bias_context = self.scene_context_registry.get_tensor(
                scene_name=scene_name,
                key_name=f'{self.key_name}.bias',
                shape=self.normalized_shape,
                device=x.device,
                dtype=x.dtype,
            )
            weight = self.weight.to(device=x.device, dtype=x.dtype) * weight_context
            bias = self.bias.to(device=x.device, dtype=x.dtype) * bias_context
        elif self.elementwise_affine:
            weight = self.weight.to(device=x.device, dtype=x.dtype)
            bias = self.bias.to(device=x.device, dtype=x.dtype)
        return F.layer_norm(x, self.normalized_shape, weight=weight, bias=bias, eps=self.eps)


class PSPLinear(PSPContextModule):
    def __init__(self, in_features, out_features, bias=True, key_name='psp.linear', seed=20260619, enabled=True):
        super().__init__(key_name=key_name, seed=seed, enabled=enabled)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear, key_name, seed=20260619, enabled=True):
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            key_name=key_name,
            seed=seed,
            enabled=enabled,
        )
        module.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.data)
        return module

    def forward(self, x):
        if self.scene_context_registry.enabled:
            scene_name = _validate_scene_name(self.active_scene_context)
            context = self.scene_context_registry.get_vector(
                scene_name=scene_name,
                key_name=self.key_name,
                size=self.in_features,
                device=x.device,
                dtype=x.dtype,
            )
            x = x * _broadcast_last_dim(context, x)
        return F.linear(x, self.weight, self.bias)


class PSPConv2d(PSPContextModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', key_name='psp.conv', seed=20260619, enabled=True):
        super().__init__(key_name=key_name, seed=seed, enabled=enabled)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = int(groups)
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_conv2d(cls, conv, key_name, seed=20260619, enabled=True):
        module = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            key_name=key_name,
            seed=seed,
            enabled=enabled,
        )
        module.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            module.bias.data.copy_(conv.bias.data)
        return module

    def forward(self, x):
        weight = self.weight
        if self.scene_context_registry.enabled:
            scene_name = _validate_scene_name(self.active_scene_context)
            context = self.scene_context_registry.get_tensor(
                scene_name=scene_name,
                key_name=self.key_name,
                shape=weight.shape[1:],
                device=weight.device,
                dtype=weight.dtype,
            )
            weight = weight * context.unsqueeze(0)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PSPMultiheadAttention(PSPContextModule):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True, key_name='psp.mha', seed=20260619, enabled=True):
        super().__init__(key_name=key_name, seed=seed, enabled=enabled)
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim={embed_dim} must be divisible by num_heads={num_heads}')
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.batch_first = bool(batch_first)
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = PSPLinear(
            self.embed_dim,
            self.embed_dim,
            bias=bias,
            key_name=f'{self.key_name}.q_proj',
            seed=seed,
            enabled=enabled,
        )
        self.k_proj = PSPLinear(
            self.embed_dim,
            self.embed_dim,
            bias=bias,
            key_name=f'{self.key_name}.k_proj',
            seed=seed,
            enabled=enabled,
        )
        self.v_proj = PSPLinear(
            self.embed_dim,
            self.embed_dim,
            bias=bias,
            key_name=f'{self.key_name}.v_proj',
            seed=seed,
            enabled=enabled,
        )
        self.out_proj = PSPLinear(
            self.embed_dim,
            self.embed_dim,
            bias=bias,
            key_name=f'{self.key_name}.out_proj',
            seed=seed,
            enabled=enabled,
        )

    def set_scene_context(self, scene_name):
        super().set_scene_context(scene_name)
        self.q_proj.set_scene_context(scene_name)
        self.k_proj.set_scene_context(scene_name)
        self.v_proj.set_scene_context(scene_name)
        self.out_proj.set_scene_context(scene_name)

    @classmethod
    def from_multihead_attention(cls, mha, key_name, seed=20260619, enabled=True):
        module = cls(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            batch_first=mha.batch_first,
            key_name=key_name,
            seed=seed,
            enabled=enabled,
        )
        q_weight, k_weight, v_weight = mha.in_proj_weight.chunk(3, dim=0)
        module.q_proj.weight.data.copy_(q_weight.data)
        module.k_proj.weight.data.copy_(k_weight.data)
        module.v_proj.weight.data.copy_(v_weight.data)
        if mha.in_proj_bias is not None:
            q_bias, k_bias, v_bias = mha.in_proj_bias.chunk(3, dim=0)
            module.q_proj.bias.data.copy_(q_bias.data)
            module.k_proj.bias.data.copy_(k_bias.data)
            module.v_proj.bias.data.copy_(v_bias.data)
        module.out_proj.weight.data.copy_(mha.out_proj.weight.data)
        if mha.out_proj.bias is not None:
            module.out_proj.bias.data.copy_(mha.out_proj.bias.data)
        return module

    def _shape_attn_mask(self, attn_mask, batch_size, target_len, source_len, dtype, device):
        if attn_mask is None:
            return None
        if attn_mask.dtype == torch.bool:
            mask = torch.zeros_like(attn_mask, dtype=dtype, device=device)
            mask = mask.masked_fill(attn_mask.to(device=device), float('-inf'))
        else:
            mask = attn_mask.to(device=device, dtype=dtype)
        if mask.dim() == 2:
            return mask.view(1, 1, target_len, source_len)
        if mask.dim() == 3:
            if mask.shape[0] == batch_size * self.num_heads:
                return mask.view(batch_size, self.num_heads, target_len, source_len)
            if mask.shape[0] == batch_size:
                return mask.view(batch_size, 1, target_len, source_len)
        raise RuntimeError(f'Unsupported attn_mask shape for PSPMultiheadAttention: {tuple(mask.shape)}')

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
        **kwargs,
    ):
        if kwargs:
            unsupported = ', '.join(sorted(kwargs.keys()))
            raise RuntimeError(f'Unsupported PSPMultiheadAttention kwargs: {unsupported}')

        is_batched = query.dim() == 3
        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, target_len, _ = query.shape
        source_len = key.shape[1]

        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)

        q_proj = q_proj.view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = k_proj.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = v_proj.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * (self.head_dim ** -0.5)

        shaped_mask = self._shape_attn_mask(
            attn_mask,
            batch_size=batch_size,
            target_len=target_len,
            source_len=source_len,
            dtype=attn_scores.dtype,
            device=attn_scores.device,
        )
        if shaped_mask is not None:
            attn_scores = attn_scores + shaped_mask

        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2:
                raise RuntimeError('key_padding_mask for PSPMultiheadAttention must be 2D')
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.to(device=attn_scores.device, dtype=torch.bool).view(batch_size, 1, 1, source_len),
                float('-inf'),
            )

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(target_len, source_len, device=attn_scores.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask.view(1, 1, target_len, source_len), float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        if self.training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_output = torch.matmul(attn_weights, v_proj)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, target_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        if not is_batched:
            attn_output = attn_output.squeeze(1)

        if not need_weights:
            return attn_output, None

        attn_weights_out = attn_weights.mean(dim=1) if average_attn_weights else attn_weights
        if not is_batched and attn_weights_out is not None:
            attn_weights_out = attn_weights_out.squeeze(0)
        return attn_output, attn_weights_out


def convert_module_to_psp(module, prefix, seed=20260619, enabled=True):
    for name, child in list(module.named_children()):
        key_name = f'{prefix}.{name}' if prefix else name
        if isinstance(child, nn.Linear):
            replacement = PSPLinear.from_linear(child, key_name=key_name, seed=seed, enabled=enabled)
            setattr(module, name, replacement)
        elif isinstance(child, nn.Conv2d):
            replacement = PSPConv2d.from_conv2d(child, key_name=key_name, seed=seed, enabled=enabled)
            setattr(module, name, replacement)
        elif isinstance(child, nn.LayerNorm):
            replacement = PSPLayerNorm.from_layer_norm(child, key_name=key_name, seed=seed, enabled=enabled)
            setattr(module, name, replacement)
        elif isinstance(child, nn.MultiheadAttention):
            replacement = PSPMultiheadAttention.from_multihead_attention(child, key_name=key_name, seed=seed, enabled=enabled)
            setattr(module, name, replacement)
        else:
            convert_module_to_psp(child, key_name, seed=seed, enabled=enabled)
    return module


def set_psp_scene_context(module, scene_name):
    for child in module.modules():
        if hasattr(child, 'set_scene_context'):
            child.set_scene_context(scene_name)
