"""Curated player entity resolution: alias -> canonical name."""

from pathlib import Path

import pandas as pd

from badminton_vision.errors import DataContractError


def load_player_aliases(path: Path) -> dict[str, str]:
    """Load the curated alias table; reject duplicate or self-referential aliases."""
    table = pd.read_csv(path, comment="#")
    aliases: dict[str, str] = {}
    for alias, canonical in zip(table["alias"], table["canonical"], strict=True):
        if alias in aliases:
            raise DataContractError(f"duplicate alias {alias!r} in {path.name}")
        if alias == canonical:
            raise DataContractError(f"self-referential alias {alias!r} in {path.name}")
        aliases[str(alias)] = str(canonical)
    if set(aliases) & set(aliases.values()):
        raise DataContractError(f"{path.name}: a canonical name is also used as an alias")
    return aliases
