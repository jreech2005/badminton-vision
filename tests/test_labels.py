"""Label-map policy: versioned contract, fail-loud on new vocabulary, unknown handling."""

from pathlib import Path

import pandas as pd
import pytest

from badminton_vision import paths
from badminton_vision.errors import LabelMapError
from badminton_vision.ingest.labels import LabelMap, apply_labels

pytestmark = pytest.mark.unit

# configs/*.yaml are versioned code artifacts, not data; unit tests may read them.
REAL_YAML = paths.CONFIGS_DIR / "shot_labels_v1.yaml"


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_real_yaml_has_18_classes_and_unknown() -> None:
    label_map = LabelMap.load(REAL_YAML)
    assert label_map.version == "shot_labels_v1"
    assert len(label_map.class_to_group) == 18
    assert label_map.unknown_tokens == frozenset({"未知球種"})


def test_both_passive_drop_spellings_map_to_one_class() -> None:
    label_map = LabelMap.load(REAL_YAML)
    # Data spells it 過度切球, the READMEs 過渡切球; both must resolve identically.
    assert label_map.canonical("過度切球") == "passive_drop"
    assert label_map.canonical("過渡切球") == "passive_drop"
    assert label_map.group("passive_drop") == "transition"


def test_unmapped_token_raises() -> None:
    label_map = LabelMap.load(REAL_YAML)
    with pytest.raises(LabelMapError, match="new vocabulary"):
        label_map.canonical("不存在的球種")


def test_unknown_token_maps_to_none() -> None:
    label_map = LabelMap.load(REAL_YAML)
    assert label_map.canonical("未知球種") is None


def test_duplicate_token_rejected(tmp_path: Path) -> None:
    bad = _write_yaml(
        tmp_path / "labels.yaml",
        """
version: test
unknown_tokens: []
classes:
  a: {zh: [殺球], group: attack}
  b: {zh: [殺球], group: net}
""",
    )
    with pytest.raises(LabelMapError, match="more than once"):
        LabelMap.load(bad)


def test_empty_classes_rejected(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path / "labels.yaml", "version: test\nunknown_tokens: []\nclasses: {}\n")
    with pytest.raises(LabelMapError, match="no classes"):
        LabelMap.load(bad)


def test_apply_labels_marks_unknown_and_groups() -> None:
    label_map = LabelMap.load(REAL_YAML)
    strokes = pd.DataFrame({"type": ["殺球", "未知球種", "發短球"]})
    labelled = apply_labels(strokes, label_map)
    # pandas represents the unknown's None as NaN; assert null-ness, not identity.
    shot_class = labelled["shot_class"].tolist()
    shot_group = labelled["shot_group"].tolist()
    assert shot_class[0] == "smash"
    assert shot_class[2] == "short_service"
    assert pd.isna(shot_class[1])
    assert shot_group[0] == "attack"
    assert shot_group[2] == "serve"
    assert pd.isna(shot_group[1])
    assert list(labelled["is_unknown"]) == [False, True, False]
