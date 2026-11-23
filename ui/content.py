"""Static page content: how-to steps and FAQ."""

from __future__ import annotations

STEPS = [
    ("Add a recording", "Upload an audio or video file, record with the microphone, or pick a demo clip. Speech of "
     "5 s or more works best."),
    ("Wait for the three models", "Each model scores the whole file and then every 2-second window. A progress bar "
     "shows the steps; a minute of audio takes seconds on the GPU, or about 40 s to 3 min on a laptop CPU."),
    ("Read the verdict, then the evidence", "REAL or FAKE, how often verdicts like it were right on test data, and "
     "which seconds pushed the score. Listen to flagged segments before you decide anything."),
]

FAQ = [
    ("Can I rely on one verdict?",
     "No. It is a screening hint. Use the trust panel under the verdict: a score near 50% or models that disagree "
     "mean the verdict is weak. Never use it as the only evidence about a real person."),
    ("Which files work?",
     "WAV, FLAC, MP3, OGG, M4A, AIFF, WebM and Opus audio; MP4, MOV, MKV, AVI and WebM video (the audio track is "
     "used). Up to 50 MB. Files longer than 10 minutes are cut to their first 10 minutes."),
    ("Why does my clip say 'doesn't look like speech'?",
     "Music, tones, noise, long silence or very quiet audio fail the speech checks. The verdict for such input is "
     "much less reliable."),
    ("Why is a fake called REAL?",
     "The models learnt from 2019 text-to-speech attacks. Modern voice conversion (for example RVC) and heavily "
     "processed audio are often missed; see Error analysis."),
    ("Is my audio uploaded anywhere?",
     "No. Everything runs on this computer (its GPU when available, else the CPU); files stay in memory for this browser session only."),
    ("How do I score many files?",
     "Use Batch: upload files or a ZIP and download one CSV row per file."),
    ("Can I call it from code?",
     "Yes: run `uvicorn api.main:app --port 8000`, then POST a file to `/predict`. The response includes the "
     "verdict, each model's score, flagged segments and the trust lookup."),
]
