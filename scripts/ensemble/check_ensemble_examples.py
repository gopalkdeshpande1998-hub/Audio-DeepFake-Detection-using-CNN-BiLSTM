"""End-to-end check of the merged detector on real recordings (CPU, as in the app).

Cases: real and fake clips from ASVspoof 2019 LA eval, In-the-Wild TEST speakers and the
DEEP-VOICE TEST persons; short clips (1.5 s, 3 s); long excerpts (60 s, 180 s); and
spliced recordings (real | fake | real) with a known fake interval.
Checks for every case:
  - each member scores the file on its own (three member probabilities);
  - the verdict is REAL or FAKE (never anything else), and realness + fakeness = 100%;
  - the timeline is the local analysis: not constant when there are several scored windows,
    and not equal to the file probability repeated;
  - waveform colours follow the local steps (regions are Real / Fake / No speech only);
  - for spliced clips: the share of the known fake interval flagged vs the share of the real parts.
Writes experiments/ensemble_binary/examples.md.

    python scripts/ensemble/check_ensemble_examples.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from adfd.paths import REPO_ROOT, configure_caches, load_paths

configure_caches()

from adfd.data.audio_cache import AudioCache  # noqa: E402
from adfd.ensemble import EnsembleAnalyzer  # noqa: E402
from adfd.inference import SR, load_audio  # noqa: E402

OUT = REPO_ROOT / "experiments" / "ensemble_binary" / "examples.md"


def clip_from_cache(name: str, rid: str, a: float = 0.0, b: float | None = None) -> np.ndarray:
    c = AudioCache(load_paths().cache_dir, name)
    n = c.length(rid)
    return c.read(rid, int(a * SR), n if b is None else min(n, int(b * SR))).astype(np.float32) / 32768.0


def flagged_share(res: dict, a: float, b: float) -> float:
    """Share of [a, b) covered by Fake steps."""
    tot = hit = 0.0
    for w in res["per_second"]:
        o = max(0.0, min(b, w["end_s"]) - max(a, w["start_s"]))
        tot += o
        if w["p_fake"] is not None and w["p_fake"] >= 0.5:
            hit += o
    return hit / tot if tot else float("nan")


def checks(res: dict) -> list[str]:
    bad = []
    if res["verdict"] not in ("Real", "Fake"):
        bad.append(f"verdict {res['verdict']}")
    if abs(100 * res["p_fake"] + 100 * res["p_real"] - 100) > 1e-9:
        bad.append("realness + fakeness != 100")
    if (res["p_fake"] >= 0.5) != (res["verdict"] == "Fake"):
        bad.append("verdict does not match P(fake)")
    if len(res["members"]) != 3 or not all(0 <= m["p_fake"] <= 1 for m in res["members"]):
        bad.append("member scores missing")
    p = np.array([w["p_fake"] for w in res["per_second"] if w["p_fake"] is not None])
    if res["time_resolved"]:
        if len(p) > 2 and np.ptp(p) < 1e-3:
            bad.append("flat timeline")
        if len(p) and np.allclose(p, res["p_fake"], atol=1e-6):
            bad.append("timeline equals the file score")
    if {r["label"] for r in res["regions"]} - {"Real", "Fake", "No speech"}:
        bad.append("unexpected region label")
    return bad


def main() -> None:
    paths = load_paths()
    rng = np.random.default_rng(0)
    asv = pd.read_csv(paths.manifest("asvspoof2019_la"), keep_default_na=False)
    ev = asv[(asv.split == "eval") & (asv.duration_s > 4)]
    itw = pd.read_csv(paths.manifest("in_the_wild_xd"), keep_default_na=False)
    itw = itw[(itw.split == "test") & (itw.duration_s > 4)]
    dv = pd.read_csv(paths.manifest("deep_voice_xd"), keep_default_na=False)
    dv_test = dv[dv.split == "test"].groupby("recording_id").first().reset_index()

    cases = []
    for lab, k in ((0, 3), (1, 3)):
        for r in ev[ev.label == lab].sample(k, random_state=1).itertuples():
            cases.append((f"ASVspoof eval {'real' if lab == 0 else 'fake ' + r.attack} {r.utt_id}", lab,
                          clip_from_cache("asvspoof2019_la", r.recording_id), None))
        for r in itw[itw.label == lab].sample(k, random_state=1).itertuples():
            cases.append((f"In-the-Wild test {'real' if lab == 0 else 'fake'} ({r.speaker})", lab,
                          clip_from_cache("in_the_wild", r.recording_id), None))
    for r in dv_test.itertuples():
        if r.recording_id in ("taylor-original", "trump-original", "taylor-to-trump", "trump-to-taylor"):
            cases.append((f"DEEP-VOICE test {r.recording_id} (60 s excerpt)", r.label,
                          clip_from_cache("deep_voice", r.recording_id, 60, 120), None))
    cases.append(("DEEP-VOICE trump-original (180 s excerpt, long)", 0,
                  clip_from_cache("deep_voice", "trump-original", 0, 180), None))
    rr = ev[ev.label == 0].sample(2, random_state=5)
    ff = ev[ev.label == 1].sample(1, random_state=5)
    short_src = clip_from_cache("asvspoof2019_la", rr.iloc[0].recording_id)
    cases.append(("short 1.5 s (ASVspoof real)", 0, short_src[: int(1.5 * SR)], None))
    cases.append(("short 3.0 s (ASVspoof real)", 0, short_src[: int(3.0 * SR)], None))
    fk = clip_from_cache("asvspoof2019_la", ff.iloc[0].recording_id)
    cases.append(("short 3.0 s (ASVspoof fake)", 1, fk[: int(3.0 * SR)], None))
    # spliced: real | fake | real (ASVspoof), and real | fake | real (In-the-Wild test speakers)
    gap = np.zeros(int(0.2 * SR), np.float32)
    for tag, reals, fake in (
            ("ASVspoof", [clip_from_cache("asvspoof2019_la", r) for r in rr.recording_id], fk),
            ("In-the-Wild", [clip_from_cache("in_the_wild", r) for r in
                             itw[itw.label == 0].sample(2, random_state=9).recording_id],
             clip_from_cache("in_the_wild", itw[itw.label == 1].sample(1, random_state=9).iloc[0].recording_id))):
        x = np.concatenate([reals[0], gap, fake, gap, reals[1]])
        a = (len(reals[0]) + len(gap)) / SR
        cases.append((f"spliced {tag}: real | fake | real", 1, x, (a, a + len(fake) / SR)))

    ens = EnsembleAnalyzer()
    rows, failures = [], 0
    for name, lab, x, fake_iv in cases:
        t = time.time()
        res = ens.analyze(x)
        dt = time.time() - t
        bad = checks(res)
        failures += bool(bad)
        p = [w["p_fake"] for w in res["per_second"] if w["p_fake"] is not None]
        row = {"case": name, "truth": "FAKE" if lab else "REAL", "dur_s": round(len(x) / SR, 1),
               "verdict": res["verdict"].upper(), "correct": (res["verdict"] == "Fake") == bool(lab),
               "fake_%": round(100 * res["p_fake"], 1),
               "E7a / VGi / VGv2 %": " / ".join(f"{100 * m['p_fake']:.0f}" for m in res["members"]),
               "timeline min–max %": f"{100 * min(p):.0f}–{100 * max(p):.0f}" if p else "–",
               "steps": len(res["per_second"]), "flagged segments": len(res["segments"]),
               "time_s": round(dt, 1), "checks": "ok" if not bad else "; ".join(bad)}
        if fake_iv:
            a, b = fake_iv
            row["splice: fake part flagged %"] = round(100 * flagged_share(res, a, b), 0)
            row["splice: real parts flagged %"] = round(100 * np.nanmean(
                [flagged_share(res, 0, a), flagged_share(res, b, res["duration_s"])]), 0)
            row["case"] += f" (fake {a:.1f}–{b:.1f} s)"
        rows.append(row)
        print({k: v for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    acc = df["correct"].mean() * 100
    md = ["# Merged detector: end-to-end examples (CPU)", "",
          "Produced by `scripts/ensemble/check_ensemble_examples.py`. A handful of clips, so the accuracy here is anecdotal; "
          "the dataset-level numbers are in `ensemble.md`.", "",
          f"Structural checks failed in {failures} of {len(df)} cases. Verdict correct in {acc:.0f}% of cases.", "",
          "| " + " | ".join(df.columns) + " |", "|---" * len(df.columns) + "|"]
    md += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
           for r in df.itertuples(index=False)]
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    df.to_csv(OUT.with_suffix(".csv"), index=False)
    print(f"\nwrote {OUT}; structural failures: {failures}")


if __name__ == "__main__":
    main()
