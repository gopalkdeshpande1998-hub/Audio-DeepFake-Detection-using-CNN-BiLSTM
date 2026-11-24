"""How far to trust one verdict of the app's model, WavLM + CNN-BiLSTM-Att (E7a).

Looks up the held-out accuracy of verdicts like this one (same confidence band of E7a's
calibrated P(fake)) in the error analysis written by `scripts/evaluation/error_analysis.py`
(experiments/error_analysis_e7a/error_analysis.json). Given several member scores (the retired
ensemble tables) it also matches the number of members sharing the verdict. Nothing here changes
the verdict; it only reports how often such verdicts were right on test data.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from adfd.paths import REPO_ROOT

PATH = REPO_ROOT / "experiments" / "error_analysis_e7a" / "error_analysis.json"
BINS = [0.1, 0.2, 0.3, 0.4]                     # |p - 0.5| upper edges of bands 0..3; band 4 above
LEVELS = ["very low", "low", "medium", "high", "very high"]


@lru_cache(maxsize=1)
def load(path: Path = PATH) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def band(p_fake: float) -> int:
    d = abs(float(p_fake) - 0.5)
    return next((i for i, b in enumerate(BINS) if d < b), len(BINS))


def assess(p_fake: float, member_p: list[float]) -> dict:
    """{'level', 'band', 'agree', 'n_members', 'sets': {name: {'accuracy_pct', 'n', 'basis'}}, 'advice'}."""
    member_p = [q for q in member_p if q is not None and q == q]      # failed members are not votes
    b = band(p_fake)
    fake = p_fake >= 0.5
    agree = sum((q >= 0.5) == fake for q in member_p)
    out = {"level": LEVELS[b], "band": b, "agree": agree, "n_members": len(member_p), "sets": {}}
    rep = load()
    if rep:
        for name, d in rep["sets"].items():
            rel = d["reliability"]
            cell = rel["grid"].get(f"{agree}|{b}")
            if cell:
                out["sets"][name] = {**cell, "basis": "same confidence" if len(member_p) == 1 else "same confidence and agreement"}
            else:
                conf = rel["by_confidence"].get(rep["confidence_bins"][b])
                if conf:
                    out["sets"][name] = {**conf, "basis": "same confidence"}
    if b <= 0:
        out["advice"] = ("The score is close to the 50% line: treat this as undecided and check the "
                         "recording by other means.")
    elif len(member_p) == 1:
        out["advice"] = ("The score is far from the line: the strongest kind of verdict this model gives, still not "
                         "proof." if b >= 3 else "A usable hint, not proof. Listen to the flagged segments.")
    elif agree < len(member_p) and b <= 2:
        out["advice"] = "The models disagree and the score is not far from the line: treat the verdict as weak."
    elif b >= 3 and agree == len(member_p):
        out["advice"] = "All models agree and the score is far from the line: the strongest kind of verdict here."
    else:
        out["advice"] = "A usable hint, not proof. Listen to the flagged segments."
    return out
