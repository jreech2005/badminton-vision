"""Loader contract: authoritative match.csv, 30-column strokes, fail-loud on warts."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from badminton_vision import paths
from badminton_vision.errors import DataContractError, LabelMapError
from badminton_vision.ingest.labels import LabelMap, apply_labels
from badminton_vision.ingest.shuttleset import (
    SHUTTLESET22_SPEC,
    SHUTTLESET_SPEC,
    SourceSpec,
    load_all_strokes,
    load_homography,
    load_match_table,
    load_strokes,
)
from tests.conftest import write_mini_dataset


def _spec(set_dir: Path) -> SourceSpec:
    return SourceSpec(name="mini", set_dir=set_dir, uid_prefix="mini")


@pytest.mark.unit
def test_match_table_coerces_float_dates(tmp_path: Path) -> None:
    spec = _spec(write_mini_dataset(tmp_path, float_dates=True))
    table = load_match_table(spec)
    assert len(table) == 2
    assert table.loc[0, "date"] == dt.date(2022, 3, 5)
    assert table.loc[0, "match_uid"] == "mini:1"
    assert table.loc[0, "date_precision"] == "day"


@pytest.mark.unit
def test_match_table_applies_aliases(tmp_path: Path) -> None:
    spec = _spec(write_mini_dataset(tmp_path))
    table = load_match_table(spec, aliases={"Alpha": "ALPHA Canonical"})
    assert table.loc[0, "winner"] == "ALPHA Canonical"
    assert table.loc[0, "loser"] == "Beta"


@pytest.mark.unit
def test_match_table_missing_video_dir_raises(tmp_path: Path) -> None:
    set_dir = write_mini_dataset(tmp_path)
    matches = pd.read_csv(set_dir / "match.csv")
    matches.loc[0, "video"] = "Does_Not_Exist"
    matches.to_csv(set_dir / "match.csv", index=False)
    with pytest.raises(DataContractError, match="not found verbatim"):
        load_match_table(_spec(set_dir))


@pytest.mark.unit
def test_match_table_invalid_date_raises(tmp_path: Path) -> None:
    set_dir = write_mini_dataset(tmp_path)
    matches = pd.read_csv(set_dir / "match.csv")
    matches.loc[0, "month"] = 13
    matches.to_csv(set_dir / "match.csv", index=False)
    with pytest.raises(DataContractError, match="invalid date"):
        load_match_table(_spec(set_dir))


@pytest.mark.unit
def test_strokes_reject_column_deviation(tmp_path: Path) -> None:
    set_dir = write_mini_dataset(tmp_path)
    stroke_file = set_dir / "Alpha_Beta_Open_Finals" / "set1.csv"
    strokes = pd.read_csv(stroke_file).drop(columns=["flaw"])
    strokes.to_csv(stroke_file, index=False)
    with pytest.raises(DataContractError, match="30-column contract"):
        load_strokes(_spec(set_dir), "mini:1", "Alpha_Beta_Open_Finals")


@pytest.mark.unit
def test_strokes_set_count_mismatch_raises(tmp_path: Path) -> None:
    set_dir = write_mini_dataset(tmp_path)
    with pytest.raises(DataContractError, match="set files"):
        load_strokes(_spec(set_dir), "mini:2", "Gamma_Delta_Open_SemiFinals", expected_sets=2)


@pytest.mark.unit
def test_load_all_strokes_stamps_provenance(tmp_path: Path) -> None:
    spec = _spec(write_mini_dataset(tmp_path))
    strokes = load_all_strokes(spec, load_match_table(spec))
    # 2 sets + 1 set of the 6-row synthetic frame
    assert len(strokes) == 18
    assert set(strokes["match_uid"]) == {"mini:1", "mini:2"}
    assert set(strokes.loc[strokes["match_uid"] == "mini:1", "set_no"]) == {1, 2}
    assert (strokes["source"] == "mini").all()


@pytest.mark.unit
def test_load_homography_parses_3x3(tmp_path: Path) -> None:
    spec = _spec(write_mini_dataset(tmp_path))
    matrices = load_homography(spec)
    assert set(matrices) == {"Alpha_Beta_Open_Finals", "Gamma_Delta_Open_SemiFinals"}
    assert matrices["Alpha_Beta_Open_Finals"].shape == (3, 3)


@pytest.mark.integration
def test_real_shuttleset_counts() -> None:
    table = load_match_table(SHUTTLESET_SPEC)
    strokes = load_all_strokes(SHUTTLESET_SPEC, table)
    assert len(table) == 44
    # Loader-observed stroke count; the dataset README claims 36,492 but the
    # extracted set CSVs sum to 36,484 (master doc's figure).
    assert len(strokes) == 36484
    assert set(load_homography(SHUTTLESET_SPEC)) >= set(table["video"])


@pytest.mark.integration
def test_real_shuttleset22_counts() -> None:
    table = load_match_table(SHUTTLESET22_SPEC)
    strokes = load_all_strokes(SHUTTLESET22_SPEC, table)
    assert len(table) == 58
    assert len(strokes) == 52356
    assert set(load_homography(SHUTTLESET22_SPEC)) >= set(table["video"])


@pytest.mark.integration
def test_real_label_coverage_and_unknown_counts() -> None:
    label_map = LabelMap.load(paths.CONFIGS_DIR / "shot_labels_v1.yaml")
    unknown_expected = {"shuttleset": 1407, "shuttleset22": 1078}
    for spec in (SHUTTLESET_SPEC, SHUTTLESET22_SPEC):
        strokes = load_all_strokes(spec, load_match_table(spec))
        try:
            labelled = apply_labels(strokes, label_map)
        except LabelMapError as exc:  # pragma: no cover - failure formatting only
            pytest.fail(f"{spec.name}: real vocabulary not covered: {exc}")
        assert int(labelled["is_unknown"].sum()) == unknown_expected[spec.name]
