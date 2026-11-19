"""Unit tests for manifest parsing and validation (no datasets needed)."""

import pandas as pd
import pytest

from adfd.data.manifests import (
    BONAFIDE,
    COLUMNS,
    SPOOF,
    make_windows,
    parse_asvspoof_protocol,
    parse_deepvoice_name,
    parse_itw_meta,
    validate_manifest,
)


def test_parse_asvspoof_protocol():
    rows = parse_asvspoof_protocol([
        "LA_0079 LA_T_1138215 - - bonafide\n",
        "\n",
        "LA_0039 LA_E_2834763 - A11 spoof\n",
    ])
    assert rows == [
        {"speaker": "LA_0079", "utt_id": "LA_T_1138215", "attack": "-", "label": BONAFIDE},
        {"speaker": "LA_0039", "utt_id": "LA_E_2834763", "attack": "A11", "label": SPOOF},
    ]


@pytest.mark.parametrize("line", ["LA_0079 LA_T_1 - - real", "LA_0079 LA_T_1 - bonafide"])
def test_parse_asvspoof_protocol_rejects_bad_lines(line):
    with pytest.raises(ValueError):
        parse_asvspoof_protocol([line])


def test_parse_itw_meta():
    rows = parse_itw_meta([
        {"file": "0.wav", "speaker": "Alec Guinness", "label": "spoof"},
        {"file": "5.wav", "speaker": "Barack Obama", "label": "bona-fide"},
    ])
    assert [r["label"] for r in rows] == [SPOOF, BONAFIDE]
    with pytest.raises(ValueError):
        parse_itw_meta([{"file": "1.wav", "speaker": "x", "label": "real"}])


@pytest.mark.parametrize("stem, expected", [
    ("biden-original", ("real", "biden", "biden")),
    ("Obama-to-Trump", ("fake", "obama", "trump")),
    ("trump-to-Biden", ("fake", "trump", "biden")),
])
def test_parse_deepvoice_name(stem, expected):
    assert parse_deepvoice_name(stem) == expected


def test_parse_deepvoice_name_rejects_unknown():
    with pytest.raises(ValueError):
        parse_deepvoice_name("linus-original-DEMO")


def test_make_windows_drops_partial_tail():
    assert make_windows(9.5, 4.0, 2.0) == [(0.0, 4.0), (2.0, 6.0), (4.0, 8.0)]
    assert make_windows(4.0, 4.0, 2.0) == [(0.0, 4.0)]
    assert make_windows(3.9, 4.0, 2.0) == []


def _itw_like(n_real: int, n_fake: int) -> pd.DataFrame:
    n = n_real + n_fake
    return pd.DataFrame({
        "utt_id": [f"itw_{i}" for i in range(n)],
        "path": [f"{i}.wav" for i in range(n)],
        "label": [BONAFIDE] * n_real + [SPOOF] * n_fake,
        "dataset": "in_the_wild", "split": "test", "speaker": "x",
        "attack": ["-"] * n_real + ["unknown"] * n_fake,
        "start_s": 0.0, "end_s": 2.0, "duration_s": 2.0,
        "sample_rate": 16000, "channels": 1,
        "recording_id": [f"itw_{i}" for i in range(n)],
    })[COLUMNS]


def test_validate_flags_wrong_counts_and_duplicates():
    df = _itw_like(3, 2)
    problems = validate_manifest(df, "in_the_wild")
    assert any("expected (total, real, fake)" in p for p in problems)

    dup = pd.concat([df, df.iloc[:1]], ignore_index=True)
    assert any("duplicate utt_id" in p for p in validate_manifest(dup, "in_the_wild"))


def test_validate_flags_bad_segments():
    df = _itw_like(1, 1)
    df.loc[0, "end_s"] = 5.0  # past the end of a 2 s file
    assert any("ending after the end" in p for p in validate_manifest(df, "in_the_wild"))
