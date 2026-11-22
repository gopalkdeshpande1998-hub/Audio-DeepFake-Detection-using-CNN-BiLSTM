"""Audio deepfake detection REST API (FastAPI).

    uvicorn api.main:app --port 8000        # docs at http://localhost:8000/docs

Endpoints
  GET  /health    service status and the default model
  GET  /models    available models
  POST /predict   multipart audio file -> verdict, calibrated P(fake), OOD check, timeline
                  query: model=<run name> (optional; default: the merged "ensemble"), timeline=true|false
                  The ensemble also returns each member's vote, Real / Uncertain / Fake regions over
                  time and a plain-language explanation (explanation=false skips the acoustic part).
Inference runs on the CPU (set ADFD_DEVICE=cuda to use a GPU).
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from adfd.paths import configure_caches

configure_caches()

from adfd import __version__ as ADFD_VERSION  # noqa: E402
from adfd import registry  # noqa: E402
from adfd.inference import SR, load_audio  # noqa: E402

MAX_BYTES = 50 * 2**20
DEVICE = os.environ.get("ADFD_DEVICE", "cpu")

app = FastAPI(title="Audio Deepfake Detection API", version=ADFD_VERSION,
              description="Calibrated Real / Uncertain / Fake verdicts for speech recordings. Research prototype: "
                          "trained on ASVspoof 2019 LA and much less reliable on real-world audio.")


class Segment(BaseModel):
    start_s: float
    end_s: float
    p_fake: float


class Member(BaseModel):
    model: str
    label: str
    p_fake: float
    vote: str


class Region(BaseModel):
    label: str
    start_s: float
    end_s: float
    p_fake_mean: float


class Prediction(BaseModel):
    file: str
    model: str
    verdict: str
    p_fake: float
    calibrated: bool
    out_of_distribution: bool
    ood_reasons: list[str]
    duration_s: float
    suspicious_segments: list[Segment]
    timeline: list[Segment] | None = None
    members: list[Member] | None = None
    regions: list[Region] | None = None
    gate_reason: str | None = None
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
                  explanation: bool = Query(True, description="ensemble only: explanation statements")) -> Prediction:
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
    res = an.analyze(x, explain=explanation) if run == registry.ENSEMBLE else an.analyze(x)
    seg = lambda w: Segment(start_s=w["start_s"], end_s=w["end_s"], p_fake=w["p_fake"])  # noqa: E731
    ens = run == registry.ENSEMBLE
    return Prediction(file=file.filename or "upload", model=run, verdict=res["verdict"], p_fake=res["p_fake"],
                      calibrated=res["calibrated"], out_of_distribution=res["out_of_distribution"],
                      ood_reasons=res["ood_reasons"], duration_s=res["duration_s"],
                      suspicious_segments=[seg(w) for w in res["suspicious_segments"]],
                      timeline=[seg(w) for w in res["timeline"]] if timeline else None,
                      members=[Member(model=m["run"], label=m["label"], p_fake=m["p_fake"], vote=m["vote"])
                               for m in res["members"]] if ens else None,
                      regions=[Region(**{k: r[k] for k in ("label", "start_s", "end_s", "p_fake_mean")})
                               for r in res["regions"]] if ens else None,
                      gate_reason=res.get("gate_reason") if ens else None,
                      explanation=res["explanation"]["lines"] if ens and "explanation" in res else None)
