from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gaussia.core.transform import Transform

if TYPE_CHECKING:
    from configuration import LiteralTransformConfig


@dataclass(frozen=True)
class LiteralTransform(Transform):
    transform_key: str
    source: str
    target: str

    @property
    def key(self) -> str:
        return self.transform_key

    def apply(self, entity: str) -> str:
        if entity != self.source:
            raise ValueError(f"unexpected entity for {self.key}: {entity}")
        return self.target


def build_transforms(configs: list[LiteralTransformConfig]) -> tuple[Transform, ...]:
    return tuple(LiteralTransform(config.key, config.source, config.target) for config in configs)
