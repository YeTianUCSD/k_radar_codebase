"""Versioned protocol signatures for semantic-superposition checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SIGNATURE_SCHEMA = "kradar-semantic-superposition-protocol/v1"


def _jsonable(value: Any):
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"protocol signature cannot serialize {type(value).__name__}"
    )


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_protocol_signature(
    *,
    pipeline_id: str,
    checkpoint_version: int,
    experiment_config: Mapping[str, Any],
    selected_scenes: Sequence[str],
    init_model: str,
    execution_controls: Mapping[str, Any],
):
    """Build a deterministic signature for all resume-relevant V3 inputs."""
    payload = {
        "pipeline_id": str(pipeline_id),
        "checkpoint_version": int(checkpoint_version),
        "experiment_config": _jsonable(experiment_config),
        "selected_scenes": [str(name) for name in selected_scenes],
        "init_model": str(Path(init_model).expanduser().resolve()),
        "execution_controls": _jsonable(execution_controls),
    }
    return {
        "schema": SIGNATURE_SCHEMA,
        "sha256": _digest(payload),
        "payload": payload,
    }


def validate_protocol_signature(stored, expected):
    """Reject missing, corrupt, or protocol-incompatible V3 signatures."""
    if not isinstance(stored, Mapping):
        raise ValueError(
            "V3 resume checkpoint has no protocol signature; "
            "V2 and legacy checkpoints cannot be resumed as V3"
        )
    if stored.get("schema") != SIGNATURE_SCHEMA:
        raise ValueError(
            f"unsupported V3 protocol-signature schema: {stored.get('schema')!r}"
        )
    stored_payload = stored.get("payload")
    if not isinstance(stored_payload, Mapping):
        raise ValueError("V3 resume checkpoint has an invalid protocol signature")
    if stored.get("sha256") != _digest(stored_payload):
        raise ValueError("V3 resume checkpoint protocol signature is corrupt")
    if stored.get("sha256") != expected.get("sha256"):
        old_id = stored_payload.get("pipeline_id")
        new_id = expected.get("payload", {}).get("pipeline_id")
        raise ValueError(
            "V3 resume protocol mismatch: checkpoint and current run use "
            f"different configuration/signature ({old_id!r} -> {new_id!r})"
        )

