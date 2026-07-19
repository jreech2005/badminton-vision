"""Versioned shot-label mapping: raw stroke tokens to canonical classes and groups."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from badminton_vision.errors import LabelMapError


@dataclass(frozen=True)
class LabelMap:
    """Immutable token->class->group mapping loaded from one versioned YAML."""

    version: str
    unknown_tokens: frozenset[str]
    token_to_class: dict[str, str]
    class_to_group: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "LabelMap":
        """Load and validate a label-map YAML; fail loud on structural defects."""
        with path.open(encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        version = raw["version"]
        unknown_tokens = frozenset(raw["unknown_tokens"])
        token_to_class: dict[str, str] = {}
        class_to_group: dict[str, str] = {}
        for class_name, spec in raw["classes"].items():
            class_to_group[class_name] = spec["group"]
            for token in spec["zh"]:
                if token in token_to_class or token in unknown_tokens:
                    raise LabelMapError(f"{path.name}: token {token!r} mapped more than once")
                token_to_class[token] = class_name
        if not token_to_class:
            raise LabelMapError(f"{path.name}: no classes defined")
        return cls(
            version=version,
            unknown_tokens=unknown_tokens,
            token_to_class=token_to_class,
            class_to_group=class_to_group,
        )

    def canonical(self, token: str) -> str | None:
        """Map a raw token to its canonical class; None for unknown tokens."""
        if token in self.unknown_tokens:
            return None
        try:
            return self.token_to_class[token]
        except KeyError:
            raise LabelMapError(
                f"token {token!r} is not in label map {self.version}; "
                f"new vocabulary requires a new label-map version"
            ) from None

    def group(self, canonical_class: str) -> str:
        """Return the tactical group of a canonical class."""
        try:
            return self.class_to_group[canonical_class]
        except KeyError:
            raise LabelMapError(f"class {canonical_class!r} is not in {self.version}") from None


def apply_labels(strokes: pd.DataFrame, label_map: LabelMap) -> pd.DataFrame:
    """Add shot_class, shot_group, and is_unknown columns derived from `type`."""
    shot_class = strokes["type"].map(label_map.canonical)
    labelled = strokes.copy()
    labelled["shot_class"] = shot_class
    # pandas stores the Nones from canonical() as NaN, so guard with isna, not `is None`.
    labelled["shot_group"] = shot_class.map(
        lambda c: None if pd.isna(c) else label_map.group(str(c))
    )
    labelled["is_unknown"] = shot_class.isna()
    return labelled
