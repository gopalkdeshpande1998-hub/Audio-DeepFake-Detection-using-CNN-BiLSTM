"""Build and validate dataset manifests.

A manifest is a CSV with one row per scored unit (an utterance, or a window of
a long recording) and these common columns:

    utt_id        unique id within the dataset
    path          audio file, relative to the dataset root (POSIX separators)
    label         0 = bona fide (real), 1 = spoof (fake)
    dataset       asvspoof2019_la | in_the_wild | deep_voice
    split         train | dev | eval (ASVspoof) or test (cross-dataset sets)
    speaker       speaker id; for DEEP-VOICE fakes, the voice that is heard (conversion target)
    attack        ASVspoof attack id (A01-A19), "rvc" for DEEP-VOICE fakes, "-" for real
    start_s       segment start in seconds (0 for whole files)
    end_s         segment end in seconds (the file duration for whole files)
    duration_s    duration of the whole file in seconds
    sample_rate   file sample rate (the loader resamples to 16 kHz)
    channels      file channel count (the loader downmixes to mono)
    recording_id  file stem; groups the windows of one recording

Labels come only from the official metadata (ASVspoof protocol files,
In-the-Wild meta.csv, DEEP-VOICE REAL/FAKE folders), so stray files in the
dataset folders are ignored.
"""

from __future__ import annotations

import csv
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd
import soundfile as sf

COLUMNS = [
    "utt_id", "path", "label", "dataset", "split", "speaker", "attack",
    "start_s", "end_s", "duration_s", "sample_rate", "channels", "recording_id",
]

BONAFIDE, SPOOF = 0, 1

# Counts verified against the copies on disk (and the official dataset releases).
ASVSPOOF_EXPECTED = {  # split: (total, bona fide, spoof)
    "train": (25380, 2580, 22800),
    "dev": (24844, 2548, 22296),
    "eval": (71237, 7355, 63882),
}
ASVSPOOF_ATTACKS = {
    "train": {f"A{i:02d}" for i in range(1, 7)},
    "dev": {f"A{i:02d}" for i in range(1, 7)},
    "eval": {f"A{i:02d}" for i in range(7, 20)},
}
ITW_EXPECTED = (31779, 19963, 11816)
DEEPVOICE_EXPECTED_RECORDINGS = {BONAFIDE: 8, SPOOF: 56}

DEEPVOICE_WINDOW_S = 4.0
DEEPVOICE_HOP_S = 2.0


# ----------------------------------------------------------------------------
# Pure parsing helpers (unit-tested without the datasets)
# ----------------------------------------------------------------------------

def parse_asvspoof_protocol(lines: Iterable[str]) -> list[dict]:
    """Parse ASVspoof 2019 LA CM protocol lines: SPEAKER UTT - ATTACK LABEL."""
    rows = []
    for n, line in enumerate(lines, 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"protocol line {n}: expected 5 fields, got {len(parts)}: {line!r}")
        speaker, utt, _, attack, key = parts
        if key not in ("bonafide", "spoof"):
            raise ValueError(f"protocol line {n}: unknown label {key!r}")
        rows.append({
            "speaker": speaker,
            "utt_id": utt,
            "attack": attack,
            "label": BONAFIDE if key == "bonafide" else SPOOF,
        })
    return rows


def parse_itw_meta(rows: Iterable[dict]) -> list[dict]:
    """Parse In-the-Wild meta.csv rows: file, speaker, label (bona-fide | spoof)."""
    out = []
    for r in rows:
        key = r["label"].strip().lower()
        if key not in ("bona-fide", "spoof"):
            raise ValueError(f"meta.csv: unknown label {r['label']!r} for {r['file']!r}")
        out.append({
            "file": r["file"].strip(),
            "speaker": r["speaker"].strip(),
            "label": BONAFIDE if key == "bona-fide" else SPOOF,
        })
    return out


_DV_FAKE = re.compile(r"^(?P<src>[a-z]+)-to-(?P<tgt>[a-z]+)$")
_DV_REAL = re.compile(r"^(?P<spk>[a-z]+)-original$")


def parse_deepvoice_name(stem: str) -> tuple[str, str, str]:
    """Return (kind, source_speaker, target_speaker) for a DEEP-VOICE file stem.

    'biden-original' -> ('real', 'biden', 'biden')
    'Obama-to-Trump' -> ('fake', 'obama', 'trump')   # obama's speech in trump's voice
    """
    s = stem.strip().lower()
    if m := _DV_REAL.match(s):
        return "real", m["spk"], m["spk"]
    if m := _DV_FAKE.match(s):
        return "fake", m["src"], m["tgt"]
    raise ValueError(f"Unrecognised DEEP-VOICE file name: {stem!r}")


def make_windows(duration_s: float, window_s: float, hop_s: float) -> list[tuple[float, float]]:
    """Full-length windows covering a recording; a partial tail is dropped."""
    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window and hop must be positive")
    windows = []
    start = 0.0
    eps = 1e-6
    while start + window_s <= duration_s + eps:
        windows.append((round(start, 3), round(start + window_s, 3)))
        start += hop_s
    return windows


# ----------------------------------------------------------------------------
# Audio probing
# ----------------------------------------------------------------------------

def probe(files: list[Path], workers: int = 16) -> list[tuple[float, int, int]]:
    """Return (duration_s, sample_rate, channels) for each file, in order."""

    def one(p: Path) -> tuple[float, int, int]:
        info = sf.info(str(p))
        return info.frames / info.samplerate, info.samplerate, info.channels

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, files, chunksize=64))


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


# ----------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------

def build_asvspoof(root: Path, workers: int = 16) -> pd.DataFrame:
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"
    protocol_files = {
        "train": proto_dir / "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    frames = []
    for split, proto in protocol_files.items():
        with open(proto, encoding="utf-8") as f:
            rows = parse_asvspoof_protocol(f)
        files = [root / f"ASVspoof2019_LA_{split}" / "flac" / f"{r['utt_id']}.flac" for r in rows]
        info = probe(files, workers)
        frames.append(pd.DataFrame({
            "utt_id": [r["utt_id"] for r in rows],
            "path": [_rel(p, root) for p in files],
            "label": [r["label"] for r in rows],
            "dataset": "asvspoof2019_la",
            "split": split,
            "speaker": [r["speaker"] for r in rows],
            "attack": [r["attack"] for r in rows],
            "start_s": 0.0,
            "end_s": [d for d, _, _ in info],
            "duration_s": [d for d, _, _ in info],
            "sample_rate": [sr for _, sr, _ in info],
            "channels": [ch for _, _, ch in info],
            "recording_id": [r["utt_id"] for r in rows],
        }))
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def build_in_the_wild(root: Path, workers: int = 16) -> pd.DataFrame:
    with open(root / "meta.csv", encoding="utf-8", newline="") as f:
        rows = parse_itw_meta(csv.DictReader(f))
    files = [root / r["file"] for r in rows]
    info = probe(files, workers)
    ids = [Path(r["file"]).stem for r in rows]
    return pd.DataFrame({
        "utt_id": [f"itw_{i}" for i in ids],
        "path": [_rel(p, root) for p in files],
        "label": [r["label"] for r in rows],
        "dataset": "in_the_wild",
        "split": "test",
        "speaker": [r["speaker"] for r in rows],
        "attack": ["-" if r["label"] == BONAFIDE else "unknown" for r in rows],
        "start_s": 0.0,
        "end_s": [d for d, _, _ in info],
        "duration_s": [d for d, _, _ in info],
        "sample_rate": [sr for _, sr, _ in info],
        "channels": [ch for _, _, ch in info],
        "recording_id": [f"itw_{i}" for i in ids],
    })[COLUMNS]


def build_deep_voice(
    root: Path,
    window_s: float = DEEPVOICE_WINDOW_S,
    hop_s: float = DEEPVOICE_HOP_S,
    workers: int = 8,
) -> pd.DataFrame:
    recordings = []
    for folder, label in (("REAL", BONAFIDE), ("FAKE", SPOOF)):
        for p in sorted((root / folder).glob("*.wav")):
            kind, src, tgt = parse_deepvoice_name(p.stem)
            expected = "real" if label == BONAFIDE else "fake"
            if kind != expected:
                raise ValueError(f"{p}: name says {kind!r} but it is in the {folder} folder")
            recordings.append((p, label, src, tgt))
    info = probe([r[0] for r in recordings], workers)

    rows = []
    for (p, label, src, tgt), (dur, sr, ch) in zip(recordings, info):
        rec_id = p.stem.lower()
        for start, end in make_windows(dur, window_s, hop_s):
            rows.append({
                "utt_id": f"dv_{rec_id}_{int(round(start * 1000)):08d}",
                "path": _rel(p, root),
                "label": label,
                "dataset": "deep_voice",
                "split": "test",
                "speaker": tgt,
                "attack": "-" if label == BONAFIDE else "rvc",
                "start_s": start,
                "end_s": end,
                "duration_s": dur,
                "sample_rate": sr,
                "channels": ch,
                "recording_id": rec_id,
                "source_speaker": src,
                "target_speaker": tgt,
            })
    return pd.DataFrame(rows)[COLUMNS + ["source_speaker", "target_speaker"]]


BUILDERS = {
    "asvspoof2019_la": build_asvspoof,
    "in_the_wild": build_in_the_wild,
    "deep_voice": build_deep_voice,
}


# ----------------------------------------------------------------------------
# Validation and summaries
# ----------------------------------------------------------------------------

def _counts(df: pd.DataFrame) -> tuple[int, int, int]:
    return len(df), int((df["label"] == BONAFIDE).sum()), int((df["label"] == SPOOF).sum())


def validate_manifest(df: pd.DataFrame, name: str) -> list[str]:
    """Return a list of problems; an empty list means the manifest is valid."""
    problems: list[str] = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    if not df["label"].isin([BONAFIDE, SPOOF]).all():
        problems.append("labels must be 0 or 1")
    if df["utt_id"].duplicated().any():
        problems.append(f"{int(df['utt_id'].duplicated().sum())} duplicate utt_id values")
    if (df["duration_s"] <= 0).any():
        problems.append(f"{int((df['duration_s'] <= 0).sum())} files with non-positive duration")
    if (df["end_s"] <= df["start_s"]).any():
        problems.append("segments with end_s <= start_s")
    if (df["end_s"] > df["duration_s"] + 1e-3).any():
        problems.append("segments ending after the end of the file")

    if name == "asvspoof2019_la":
        for split, expected in ASVSPOOF_EXPECTED.items():
            got = _counts(df[df["split"] == split])
            if got != expected:
                problems.append(f"{split}: expected (total, real, fake) = {expected}, got {got}")
            attacks = set(df[(df["split"] == split) & (df["label"] == SPOOF)]["attack"])
            if attacks != ASVSPOOF_ATTACKS[split]:
                problems.append(f"{split}: unexpected attack set {sorted(attacks)}")
        if set(df["split"]) != set(ASVSPOOF_EXPECTED):
            problems.append(f"unexpected splits {sorted(set(df['split']))}")
        if (df["sample_rate"] != 16000).any():
            problems.append("ASVspoof files that are not 16 kHz")
        # The same speaker id must not appear in train and in dev/eval.
        train_spk = set(df[df["split"] == "train"]["speaker"])
        for split in ("dev", "eval"):
            overlap = train_spk & set(df[df["split"] == split]["speaker"])
            if overlap:
                problems.append(f"speakers shared by train and {split}: {sorted(overlap)[:5]}")

    elif name == "in_the_wild":
        if _counts(df) != ITW_EXPECTED:
            problems.append(f"expected (total, real, fake) = {ITW_EXPECTED}, got {_counts(df)}")
        if set(df["split"]) != {"test"}:
            problems.append("In-the-Wild must be test-only")
        if (df["sample_rate"] != 16000).any():
            problems.append("In-the-Wild files that are not 16 kHz")

    elif name == "deep_voice":
        recs = df.drop_duplicates("recording_id")
        for label, expected in DEEPVOICE_EXPECTED_RECORDINGS.items():
            got = int((recs["label"] == label).sum())
            if got != expected:
                problems.append(f"expected {expected} recordings with label {label}, got {got}")
        if (df.groupby("recording_id")["label"].nunique() > 1).any():
            problems.append("a recording has windows with different labels")
        lengths = (df["end_s"] - df["start_s"]).round(3).unique()
        if len(lengths) != 1:
            problems.append(f"windows of different lengths: {sorted(lengths)[:5]}")
        if set(df["split"]) != {"test"}:
            problems.append("DEEP-VOICE must be test-only")
    else:
        problems.append(f"unknown dataset name {name!r}")

    return problems


def summarise(df: pd.DataFrame) -> dict:
    """Per-split counts and hours, for the manifest summary file."""
    out = {}
    for split, part in df.groupby("split"):
        seg_hours = float((part["end_s"] - part["start_s"]).sum() / 3600)
        total, real, fake = _counts(part)
        out[split] = {
            "rows": total,
            "bonafide": real,
            "spoof": fake,
            "recordings": int(part["recording_id"].nunique()),
            "hours": round(seg_hours, 2),
            "sample_rates": sorted(int(s) for s in part["sample_rate"].unique()),
        }
    return out


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
