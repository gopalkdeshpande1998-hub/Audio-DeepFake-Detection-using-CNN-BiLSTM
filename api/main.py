"""Audio deepfake detection REST API (FastAPI).

    uvicorn api.main:app --port 8000        # docs at http://localhost:8000/docs

Endpoints
  GET  /health    service status and the default model
  GET  /models    available models
  POST /predict   multipart audio file -> verdict ("Real" or "Fake" = SYNTHETIC), calibrated P(fake),
                  P(real) = 1 - P(fake), confidence = max(P(fake), P(real)), OOD warning, timeline
                  One model decides: WavLM + CNN-BiLSTM-Att (E7a), run e7a_wavlm_crnn_s42 (adfd.single).
                  query: model=<run name> (optional; only that model is offered), timeline=true|false
                  Also returns Real / Fake / No speech regions from the time-local analysis, the flagged
                  segments with severity and evidence, and a plain-language explanation
                  (explanation=false skips the acoustic part), and a trust block: the held-out accuracy
                  of verdicts with the same confidence, from scripts/error_analysis.py.
                  Removed on 2026-11-23 with the three-model ensemble: `members` (per-model scores)
                  and the ensemble decision fields; trust.models_agreeing / n_members are always 1.
Inference runs on the GPU when one is available (CUDA, or MPS on Apple silicon Macs), else the CPU
(override: ADFD_DEVICE=cpu|cuda|mps).
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from adfd.paths import configure_caches

configure_caches()

from adfd import __version__ as ADFD_VERSION  # noqa: E402
from adfd import registry, trust  # noqa: E402
from adfd.inference import SR, load_audio  # noqa: E402

MAX_BYTES = 50 * 2**20


def _device() -> str:
    """ADFD_DEVICE if set, else the GPU PyTorch can see (CUDA, or MPS on Apple silicon), else the CPU."""
    from adfd.device import best_device

    return best_device()


DEVICE = _device()

app = FastAPI(title="Audio Deepfake Detection API", version=ADFD_VERSION,
              description="Calibrated two-way Real / Fake verdicts for speech recordings, with time-local analysis. "
                          "Research prototype: every verdict can be wrong.")


class Segment(BaseModel):
    start_s: float
    end_s: float
    p_fake: float | None          # None: no speech in this stretch (not scored)


class Region(BaseModel):
    label: str                    # "Real", "Fake" or "No speech"
    start_s: float
    end_s: float
    p_fake_mean: float | None


class FlaggedSegment(BaseModel):
    start_s: float
    end_s: float
    duration_s: float
    p_fake: float
    p_fake_peak: float
    severity: str                 # Low / Medium / High
    model_evidence: list[str] | None = None
    acoustic_evidence: list[str] | None = None


class Trust(BaseModel):
    level: str                    # very low / low / medium / high / very high (distance from the 50% line)
    models_agreeing: int          # always 1 (one model); kept for API compatibility
    n_members: int
    advice: str
    heldout_accuracy_pct: dict[str, float]   # test set -> accuracy of verdicts like this one


class Prediction(BaseModel):
    file: str
    model: str
    verdict: str                  # "Real" or "Fake" (shown as SYNTHETIC in the app)
    p_fake: float
    p_real: float
    confidence: float             # max(p_fake, p_real): the model's probability for its own call, not an accuracy
    model_label: str
    calibrated: bool
    out_of_distribution: bool
    ood_reasons: list[str]
    duration_s: float
    suspicious_segments: list[Segment]
    timeline: list[Segment] | None = None
    regions: list[Region] | None = None
    segments: list[FlaggedSegment] | None = None
    temporal: dict | None = None             # segment model: regions + 0.16-s frames (adfd.temporal)
    gate_reason: str | None = None
    trust: Trust | None = None
    explanation: list[str] | None = None


@lru_cache(maxsize=4)
def analyzer(run: str):
    return registry.make_analyzer(run, DEVICE)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": ADFD_VERSION, "default_model": registry.default_run().name,
            "device": DEVICE}


@app.get("/models")
def models() -> dict:
    return {"default": registry.default_run().name, "models": registry.available()}


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True)
async def predict(file: UploadFile = File(...), model: str | None = Query(None, description="run name, see /models"),
                  timeline: bool = Query(False, description="include the per-window scores"),
                  explanation: bool = Query(True, description="acoustic explanation statements")) -> Prediction:
    run = model or registry.default_run().name
    if run not in registry.available():
        raise HTTPException(404, f"unknown model {run!r}; see /models")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file larger than 50 MB")
    try:
        x = load_audio(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(415, f"could not decode audio ({e.__class__.__name__})") from e
    if len(x) < SR // 4:
        raise HTTPException(422, "audio shorter than 0.25 s")
    an = analyzer(run)
    res = an.analyze(x, explain=explanation)
    seg = lambda w: Segment(start_s=w["start_s"], end_s=w["end_s"], p_fake=w["p_fake"])  # noqa: E731

    def flagged(s):
        ev = s.get("evidence") or {}
        return FlaggedSegment(**{k: s[k] for k in ("start_s", "end_s", "duration_s", "p_fake", "p_fake_peak",
                                                    "severity")},
                              model_evidence=ev.get("model"),
                              acoustic_evidence=[a["statement"] for a in ev["acoustic"]] if "acoustic" in ev else None)

    t = trust.assess(res["p_fake"], [res["p_fake"]])
    trust_out = Trust(level=t["level"], models_agreeing=t["agree"], n_members=t["n_members"], advice=t["advice"],
                      heldout_accuracy_pct={k: v["accuracy_pct"] for k, v in t["sets"].items()})
    p = float(res["p_fake"])
    return Prediction(file=file.filename or "upload", model=run, model_label=res.get("model_label", run),
                      verdict=res["verdict"], p_fake=p, p_real=1.0 - p, confidence=max(p, 1.0 - p),
                      calibrated=res["calibrated"], out_of_distribution=res["out_of_distribution"],
                      ood_reasons=res["ood_reasons"], duration_s=res["duration_s"],
                      suspicious_segments=[seg(w) for w in res["suspicious_segments"]],
                      timeline=[seg(w) for w in res["timeline"]] if timeline else None,
                      regions=[Region(**{k: r[k] for k in ("label", "start_s", "end_s", "p_fake_mean")})
                               for r in res["regions"]],
                      segments=[flagged(s) for s in res["segments"]],
                      gate_reason=res.get("gate_reason"), trust=trust_out,
                      explanation=res["explanation"]["lines"] if "explanation" in res else None,
                      temporal=res.get("temporal"))
