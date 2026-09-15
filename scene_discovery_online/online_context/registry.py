"""A semantic-keyed registry whose IDs are allocated in arrival order."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Tuple


SemanticKey = Tuple[str, str, str]


def normalize_key(value: Iterable[str]) -> SemanticKey:
    parts = tuple(str(item).strip().lower() for item in value)
    if len(parts) != 3:
        raise ValueError("semantic key must contain weather, road, and lighting")
    return parts


@dataclass(frozen=True)
class ContextEntry:
    context_id: int
    weather: str
    road: str
    lighting: str
    created_step: int
    observations: int = 0

    @property
    def key(self) -> SemanticKey:
        return self.weather, self.road, self.lighting


class ContextRegistry:
    """Map stable semantic triples to persistent internal context IDs."""

    def __init__(self, base_key: Iterable[str], created_step: int = 0):
        key = normalize_key(base_key)
        self._entries = {
            key: ContextEntry(0, *key, created_step=int(created_step))
        }
        self._next_id = 1

    def find(self, key: Iterable[str]) -> Optional[ContextEntry]:
        return self._entries.get(normalize_key(key))

    def get_or_create(self, key: Iterable[str], step: int):
        normalized = normalize_key(key)
        existing = self._entries.get(normalized)
        if existing is not None:
            return existing, False
        entry = ContextEntry(self._next_id, *normalized, created_step=int(step))
        self._next_id += 1
        self._entries[normalized] = entry
        return entry, True

    def __len__(self):
        return len(self._entries)

    def entries(self):
        return tuple(sorted(self._entries.values(), key=lambda item: item.context_id))

    def to_records(self):
        return [asdict(entry) for entry in self.entries()]
