# Implementation Plan: Audio Deepfake Detection v2

This plan turns the current project (CNN-BiLSTM on MFCCs, trained on LJSpeech vs WaveFake-MelGAN) into a reproducible research system that generalises across datasets, with explainable results and a deployable app.

It covers four areas: **research gaps**, **missing basics**, **feature enhancements**, and **new features and architectures**. Work is split into eight phases. Each phase lists its tasks, the files it produces, and a "done when" check.

---

## 0. Starting point

### 0.1 Datasets on disk

| Role | Dataset | Location | Contents | Format | Labels from |
|---|---|---|---|---|---|
| **Main** (train / validation / test) | ASVspoof 2019 LA | `G:\ASVspoof2019-dataset` | train 25,380 (2,580 real / 22,800 fake), dev 24,844 (2,548 / 22,296), eval 71,237 (7,355 / 63,882) | 16 kHz mono FLAC | `ASVspoof2019_LA_cm_protocols\*.txt` |
| Cross-dataset test 1 | In-the-Wild | `G:\release_in_the_wild-dataset` | 31,779 clips: 19,963 real / 11,816 fake, 54 speakers | 16 kHz mono float WAV | `meta.csv` (`bona-fide` / `spoof`) |
| Cross-dataset test 2 | DEEP-VOICE | `G:\DEEPVOICE-dataset\KAGGLE\AUDIO` | 8 real + 56 RVC voice-converted recordings, about 8.3 h | 44.1/48 kHz stereo WAV | folder name (`REAL` / `FAKE`) |

**Notes from inspecting the data**
- **ASVspoof attacks:** train and dev use attacks A01–A06. Eval uses A07–A19, mostly unseen; A16 and A19 reuse A04 and A06. The eval set therefore already tests generalisation to new attacks.
- **Extra files:** the ASVspoof dev and eval folders contain extra `LA_*_A*` files (for example 696 in eval). These are speaker-verification enrolment recordings, not part of the anti-spoofing task. **The protocol files are the only source of truth**, so these files are ignored automatically.
- **min t-DCF:** `ASVspoof2019_LA_asv_scores` contains the official speaker-verification scores needed to compute min t-DCF.
- **DEEP-VOICE preprocessing:** the recordings are long, stereo and at 44.1/48 kHz. They must be downmixed to mono, resampled to 16 kHz and cut into windows. `DATASET-balanced.csv` holds features someone else pre-extracted with a different recipe, so it is **not used**. The `DEMONSTRATION\*.mp3` files make good demo clips for the app.
- **Test sets stay test sets:** In-the-Wild and DEEP-VOICE are **never used for training, tuning, threshold selection or augmentation**. They are only scored, once per final model.

### 0.2 Hardware and constraints

| Resource | Value | Effect on the plan |
|---|---|---|
| GPU | RTX 3050 laptop, **4 GB VRAM** | Small batches, mixed precision, base-size pretrained models only (see 4.3) |
| TensorFlow GPU | **Not available** (TF ≥ 2.11 has no GPU support on native Windows; confirmed `list_physical_devices('GPU') == []`) | **Move training to PyTorch**, which supports CUDA on native Windows |
| RAM | 16 GB | Stream audio from disk; never load a whole dataset into memory as the old notebook did |
| Disk | C: 20 GB free, G: 125 GB free | Put every cache (pip, Hugging Face, features, checkpoints) on **G:** |
| CPU | i7-12650H, 10 cores | Parallel feature extraction and data loading |

### 0.3 Main decision: switch to PyTorch

- **Why:** GPU training is impossible with the current TensorFlow 2.15 on Windows. Pretrained speech models (WavLM, wav2vec 2.0) and the reference AASIST and RawNet2 code are all PyTorch. Moving off TensorFlow also removes the protobuf conflict that forced the `starlette==1.7.0` pin, so Streamlit can be upgraded.
- **How:** keep the current TensorFlow app working until Phase 5, then switch the app to the new PyTorch models. Re-implement the CNN-BiLSTM in PyTorch layer for layer, so the paper's model stays the reference point.

---

## 1. Research question and proposed contribution

> **Research question:** can a CNN-BiLSTM detector built on pretrained self-supervised speech features, trained with codec and noise augmentation on ASVspoof 2019 LA, generalise to real-world deepfakes (In-the-Wild) and modern voice conversion (DEEP-VOICE, RVC)? And can its decisions be explained at segment level?

**Proposed model, "SSL-CNN-BiLSTM-Att":** a WavLM-Base+ front end (weighted sum of its layers) → light CNN → BiLSTM → attentive statistics pooling → classifier. It keeps the paper's CNN-BiLSTM identity and adds the ingredients currently known to generalise best.

**What makes this stand out**
1. An honest, leak-free evaluation, which the original project does not have.
2. Results on one in-domain test set and **two cross-dataset test sets**, including a modern voice-conversion dataset that most papers don't test on.
3. A clear ablation of input features, architecture, pretrained front end and augmentation.
4. A check for the known ASVspoof **silence shortcut**: the length of leading silence alone predicts the label (Müller et al., 2021).
5. Segment-level explanations plus calibrated, uncertainty-aware output in a deployable app.

---

## 2. Phases at a glance

| Phase | Theme | Main output | Rough effort |
|---|---|---|---|
| P0 | Foundations: repo, environment, data lists | `src/`, configs, manifests, PyTorch environment | 3–4 days |
| P1 | Honest baseline | CNN-BiLSTM in PyTorch with EER/min t-DCF on all 3 test sets | 4–5 days |
| P2 | Research gaps: features, shortcuts, augmentation, robustness | Ablation tables | 1.5–2 weeks |
| P3 | New architectures | AASIST, RawNet2, SSL-CNN-BiLSTM-Att, ensemble | 2 weeks |
| P4 | Reliability and explainability | Calibration, out-of-distribution check, Grad-CAM, segment timeline | 1 week |
| P5 | App feature upgrade | New Streamlit app on PyTorch | 1 week |
| P6 | Deployment and missing basics | API, Docker, tests/CI, model card, README | 4–5 days |
| P7 | Paper and release | Results tables, figures, paper draft, model release | 1–2 weeks |

Do the phases in order: P2 and P3 depend on P0/P1, and P5 depends on P3/P4. Within P2 and P3, the individual experiments are independent and can run in any order.

---

## Phase 0: Foundations

**Goal:** a reproducible project layout that any later experiment can plug into.

### 0.A Repository restructure
```
configs/                # one YAML per experiment (data, features, model, training, seed)
src/adfd/
  data/                 # manifests.py, datasets.py (streaming), crops.py, augment.py
  features/             # mfcc.py, lfcc.py, cqcc.py, logmel.py (all GPU-capable via torchaudio)
  models/               # cnn_bilstm.py, aasist.py, rawnet2.py, lcnn.py, ssl_frontend.py, heads.py
  train.py  evaluate.py  metrics.py (EER, min t-DCF, AUC)  calibrate.py  explain.py
scripts/                # build_manifests.py, run_matrix.py, make_tables.py, make_figures.py
app/                    # Streamlit UI (moved from app.py)
api/                    # FastAPI service
tests/                  # pytest
docs/                   # this plan, model card, dataset notes
experiments/            # per-run metrics and summaries, aggregated tables, figures (small; tracked in git)
                        # (not "results/": on Windows that is the same folder as the v1 Results/)
legacy/                 # original notebooks + TF model, kept for reference
```

### 0.B Environment
- [ ] New `.venv-torch` (Python 3.11) with `torch` + `torchaudio` (CUDA 12.x build), `transformers`, `soundfile`, `librosa`, `scikit-learn`, `pandas`, `pyyaml`, `tensorboard`, `pytest`.
- [ ] Point `HF_HOME`, `TORCH_HOME` and the pip cache at G: (C: has only 20 GB free).
- [ ] `requirements-train.txt` (pinned) separate from `requirements-app.txt`.
- [ ] Install `ffmpeg`, needed for codec augmentation and for decoding MP3 and video input.

### 0.C Data manifests (built once)
- [ ] `scripts/build_manifests.py` writes one CSV per dataset with columns `path, label (0 real / 1 fake), dataset, split, speaker, attack, duration_s, sample_rate`.
  - ASVspoof: parse the 3 protocol files and **ignore files not listed in them**.
  - In-the-Wild: parse `meta.csv` (`bona-fide` → 0, `spoof` → 1).
  - DEEP-VOICE: list the recordings, then create **4-second windows with 2-second overlap** as rows (`path, start_s, end_s, recording_id, source_speaker, target_speaker`).
- [ ] Check every manifest: files exist, label counts match the numbers in section 0.1, and no file appears in two splits.

### 0.D Reproducibility
- [ ] One seed controls Python, NumPy and torch. Save each run's config, git commit, package versions and the manifest hash next to its results.
- [ ] Log metrics to TensorBoard. Write per-utterance scores to `runs/<run_id>/scores_<testset>.csv` and metrics to `experiments/<run_id>/` so tables can be rebuilt without retraining.

**Done when:** `python scripts/build_manifests.py` produces 3 validated manifests, and `pytest` passes the manifest checks.

---

## Phase 1: Honest baseline (fixes the evaluation)

**Goal:** a trustworthy reference number for the existing architecture.

- [x] **P1.1 Port CNN-BiLSTM to PyTorch** (`models/cnn_bilstm.py`) with the same layers as the paper: 3×(Conv2D+BN+MaxPool) → 2×BiLSTM(128) → Dense(128) → output.
- [x] **P1.2 Fixed-length inputs:** random 4 s crops for training and a fixed centre crop for testing, instead of zero-padding everything to 16 s. Repeat short clips rather than padding them with zeros, which removes the length/padding shortcut.
- [x] **P1.3 Class imbalance:** weighted cross-entropy or a class-balanced sampler (ASVspoof is about 1 real to 9 fake).
- [x] **P1.4 Training loop:** AdamW, cosine learning-rate schedule, mixed precision, early stopping on **dev EER** (not accuracy, and never on a test set).
- [x] **P1.5 Evaluation script** (`evaluate.py`), one command for all test sets:
  - ASVspoof eval: overall EER, **min t-DCF** (using the official ASV scores), and **EER per attack (A07–A19)**.
  - In-the-Wild: EER, AUC, plus per-speaker EER for analysis.
  - DEEP-VOICE: EER per window and per recording (average the window scores), plus a confusion table by source/target speaker.
  - Accuracy, precision and recall at the threshold chosen on dev, to stay comparable with the paper.
- [ ] **P1.6 Reproduce the old setup once** *(blocked: needs LJSpeech and WaveFake on disk; they are not in the current dataset folders)* (LJSpeech vs MelGAN, fixed split grouped by sentence) to show how much the old 98.3% was inflated. Report it as a "legacy" row.

**Done when:** one command trains the baseline, and `experiments/<baseline run>/` holds EER and min t-DCF for ASVspoof eval, In-the-Wild and DEEP-VOICE. Expect cross-dataset EER to be **much worse** than in-domain; that gap motivates the rest of the plan.

**Status (2026-11-19): done** except P1.6. E1, 3 seeds (`experiments/e1_cnn_bilstm_mfcc/summary.md`):

| Test set | EER % | min t-DCF |
|---|---|---|
| ASVspoof 2019 LA eval | 7.97 ± 0.04 | 0.210 ± 0.014 |
| In-the-Wild | 47.32 ± 4.61 | n/a |
| DEEP-VOICE (per window) | 48.84 ± 1.32 | n/a |

In-domain is close to the official ASVspoof 2019 LFCC-GMM baseline. A17 (about 26%) and A18 (about 16%) are the hardest attacks. Both cross-dataset sets are near chance.

---

## Phase 2: Research gaps (features, shortcuts, augmentation, robustness)

Every experiment below uses the P1 training and evaluation code. Change one factor at a time, keep everything else fixed.

### P2.1 Input-feature ablation (same CNN-BiLSTM)
| Run | Front end | Why |
|---|---|---|
| F1 | MFCC-40 (baseline) | The paper's feature |
| F2 | LFCC-60 | Linear frequency scale keeps the high-frequency vocoder artifacts that the mel scale compresses |
| F3 | CQCC | Classic strong anti-spoofing feature |
| F4 | Log-mel spectrogram (80 bands) | Richer than MFCC; usual input for CNNs |
| F5 | Raw waveform (sinc layer, as in RawNet2) | Lets the model learn its own filters |

### P2.2 Shortcut checks
- [x] **Silence shortcut:** train and test with leading and trailing silence trimmed by a voice-activity detector, and compare with untrimmed audio. Also run a "silence-only" classifier as a control. If trimming changes cross-dataset EER a lot, the model was using silence.
- [x] **Length shortcut:** use the same fixed-length crops for every dataset (already enforced in P1.2) and report EER per duration bin and the score–duration correlation (`scripts/length_check.py`).
- [x] **Loudness/normalisation:** a loudness-only control classifier, plus every headline model re-scored with RMS-normalised test crops (`scripts/loudness_check.py`, `evaluate.py --normalize rms`). Normalisation is available for training too (`audio.normalize`).

### P2.3 Data augmentation (training only)
- [x] **RawBoost** (needs no extra data): linear/non-linear convolutive noise plus impulsive and stationary noise.
- [x] **Codec simulation** via libsndfile (no ffmpeg needed): MP3, Opus and 8 kHz μ-law telephone band. AAC is not supported by libsndfile and was left out.
- [x] **Gain, resampling and speed changes** (small amounts): gain ±6 dB; resampling to 8/11.025/12 kHz and back; speed 0.9–1.1× (E4d).
- [x] Ablation: no augmentation / RawBoost / codec / both / all. Chosen on a **degraded dev subset** (A01–A06, the same 10 degradations), because clean dev EER cannot reward robustness and the test sets must not be used for selection.

### P2.4 Robustness test (evaluation only)
- [x] Degrade the ASVspoof eval set with each codec, with white or pink noise at signal-to-noise ratios of 20/10/5 dB, and with telephone band-limiting. Plot EER against degradation level for the main models (`experiments/tables/robustness_eval.png`, `robustness_dev.png`).

### P2.5 Statistical rigour
- [x] Run each headline model with **3 seeds** and report mean ± std.
- [x] Bootstrap 95% confidence intervals on EER for the cross-dataset sets.

**Done when:** `experiments/tables/` holds the feature ablation, the shortcut-check table, the augmentation ablation and the robustness curves, each produced by `scripts/make_tables.py`.

**Status (2026-11-21): done.** Every row in every table has 3 seeds (42, 43, 44). Full tables: `experiments/tables/phase2.md` (`python scripts/make_tables.py`); robustness figures: `experiments/tables/robustness_eval.png` and `robustness_dev.png`. All EER in %.

| Model (CNN-BiLSTM) | dev | eval | eval min t-DCF | In-the-Wild | DEEP-VOICE (recording) | degraded eval, mean of 10 |
|---|---|---|---|---|---|---|
| E1 MFCC, untrimmed | 1.36 ± 0.47 | 7.97 ± 0.04 | 0.210 | 47.3 ± 4.6 | 52.4 ± 6.6 | 33.62 ± 3.91 |
| E2a LFCC, untrimmed (best feature on dev) | 0.04 ± 0.03 | 6.64 ± 1.52 | 0.155 | 42.5 ± 6.6 | 45.5 ± 7.1 | 32.41 ± 0.90 |
| E2b CQCC, untrimmed | 2.29 ± 0.22 | 11.00 ± 0.32 | 0.234 | 26.9 ± 2.8 | 30.1 ± 6.6 | – |
| E2c log-mel, untrimmed | 0.10 ± 0.05 | 6.48 ± 0.61 | 0.112 | 64.8 ± 8.1 | 50.0 ± 0.0 | – |
| E2d sinc (raw waveform), untrimmed | 0.08 ± 0.00 | 6.94 ± 0.20 | 0.129 | 55.4 ± 4.8 | 38.1 ± 1.0 | – |
| E3 LFCC, silence-trimmed | 0.13 ± 0.06 | 21.47 ± 0.95 | 0.355 | 57.2 ± 3.9 | 41.7 ± 7.2 | 34.11 ± 5.11 |
| **E4a + RawBoost** (chosen on degraded dev) | 0.34 ± 0.20 | 14.38 ± 1.09 | 0.344 | 47.4 ± 11.4 | 50.3 ± 13.0 | 24.51 ± 2.26 |
| E4b + codec | 0.64 ± 0.43 | 14.34 ± 2.10 | 0.353 | 59.0 ± 10.1 | 28.3 ± 7.3 | 33.76 ± 3.16 |
| E4c + RawBoost + codec | 1.90 ± 0.65 | 12.81 ± 0.56 | 0.310 | 53.5 ± 9.4 | 29.2 ± 7.2 | 24.33 ± 3.11 |
| E4d + RawBoost + codec + resample + speed | 10.95 ± 1.80 | 17.96 ± 0.29 | 0.459 | 36.4 ± 3.8 | 45.8 ± 7.2 | 24.82 ± 2.30 |

Selection (dev data only): **feature = LFCC** (lowest clean dev EER among the 5 features); **augmentation = E4a, RawBoost only** (lowest mean degraded-dev EER: E4a 18.43 ± 2.09, E4c 19.86 ± 2.27, E4d 20.35 ± 0.33, E4b 26.97, E3 29.64). With one seed E4c had won; with 3 seeds E4a leads, but the gap to E4c is smaller than one standard deviation, so the two are effectively tied.

Findings:
- **LFCC beats MFCC** on dev and eval (6.6 vs 8.0% eval EER). Log-mel and sinc are as good on clean ASVspoof but much worse on In-the-Wild. CQCC is the weakest on ASVspoof yet the best cross-dataset feature (In-the-Wild 26.9%, DEEP-VOICE 30.1%); it would not be chosen by dev EER, which shows clean dev EER is a poor guide to generalisation.
- **The silence shortcut is large.** Leading silence alone gives 15.9% eval EER. Scoring the untrimmed models on trimmed audio raises eval EER from 8.0 to 27.7% (MFCC) and from 6.6 to 22.1% (LFCC). The untrimmed eval numbers mostly measure silence.
- **Length is not a shortcut.** With fixed 4 s crops, EER changes little across duration bins on ASVspoof eval and |ρ| ≤ 0.17 between score and duration. On In-the-Wild the augmented models show a mild negative correlation (ρ ≈ −0.25 to −0.33).
- **Loudness is not a shortcut.** ASVspoof is peak-normalised (median peak 0 dBFS for both classes); a loudness-only classifier is near chance (46–55% EER). RMS-normalising the test audio changes model EER by at most 1.4 points.
- **On trimmed audio, augmentation helps a lot:** eval EER drops from 21.5 (E3) to 12.8–14.4% (E4a–E4c), and the degraded-eval mean from 34.1 to about 24.5% (E4a, E4c, E4d). Codec augmentation alone fixes codecs but not noise; RawBoost fixes noise. Adding resampling and speed (E4d) costs a lot of clean accuracy (dev 11%) but gives the best In-the-Wild EER of the trimmed models (36.4%).
- **Cross-dataset generalisation is still unsolved.** In-the-Wild stays at 27–65% and DEEP-VOICE varies widely between seeds. Phase 3 (pretrained SSL front ends) targets this.
- Training notes: E4c and E4d were stopped at epoch 22 of a 50-epoch cosine schedule (`--stop-after 22`, all seeds alike) to save time, so they may be slightly under-trained. E4d seed 42 was retrained after a crash caused by editing the code mid-run.

---

## Phase 3: New architectures

### P3.1 Published baselines (same data, same evaluation)
- [ ] **LCNN + LFCC**: the official ASVspoof-style baseline.
- [ ] **RawNet2**: works on the raw waveform.
- [ ] **AASIST**: spectro-temporal graph attention, a strong and small model (about 300k parameters). Its paper reports about 0.83% EER on ASVspoof 2019 LA eval; use that to check your implementation.

### P3.2 Upgrades to the CNN-BiLSTM
- [ ] Add squeeze-and-excitation and residual blocks to the CNN.
- [ ] Replace "last BiLSTM step" with **attentive statistics pooling** (attention-weighted mean and std over time).
- [ ] Try a lighter CNN and a BiLSTM vs BiGRU swap (the GRU version is faster).

### P3.3 Pretrained self-supervised front end (the core contribution)
- [ ] Front end: **WavLM-Base+** (main) and **wav2vec2-base / XLS-R-300M** (comparison), via Hugging Face `transformers`.
- [ ] Combine the hidden layers with a learned weighted sum before the CNN-BiLSTM-Att back end, giving **SSL-CNN-BiLSTM-Att**.
- [ ] Also try a WavLM front end with an AASIST back end, a strong reference combination from the literature.
- [ ] **Fitting it into 4 GB of VRAM:**
  - Fine-tune Base models (about 95M parameters) with mixed precision, 4 s crops, batch 8 plus gradient accumulation (effective batch 32), and gradient checkpointing.
  - Freeze the convolutional feature encoder, and the transformer for the first epoch.
  - **Large models** (XLS-R-300M, WavLM-Large) are **frozen only**: extract their features once and cache them on G:, then train the back end on the cache.
  - If memory is still tight, use 8-bit AdamW (`bitsandbytes`) or LoRA adapters on the transformer layers.

### P3.4 Fusion and ensembles
- [ ] Score-level fusion (logistic regression trained on **dev** scores) of the best spectrogram model, AASIST and the pretrained-feature model.
- [ ] Report whether the ensemble improves cross-dataset EER over the best single model.

### P3.5 Efficiency
- [ ] Report parameter count, model size, and inference speed on CPU and GPU (seconds of audio processed per second) for every model.
- [ ] Export the chosen model to ONNX, with optional INT8 quantization, for the app and API.

**Done when:** one main table lists every model's EER on the three test sets, its min t-DCF on ASVspoof, its size and its speed, with the best model chosen for the app.

**Status (2026-11-21): prototype run.** To reach a full working prototype quickly (user decision), Phase 3 was cut to about 5 h: `configs/queues/phase3_proto.yaml` trains **seed 42 only** with shortened schedules (E6b 15, RawNet2 12, WavLM 2, AASIST 3 epochs; LCNN s42 finished its full schedule), then fusion, efficiency and `experiments/tables/phase3.md` (n = 1 rows). E6a, E6c and the other SSL variants are not run yet. **Upgrade later:** delete the shortened run folders and run `phase3_p31_p32.yaml` and `phase3_p33_final.yaml` (3 seeds, full schedules). ONNX: `scripts/export_onnx.py` exports the back end (the TorchScript exporter cannot export the complex STFT of the spectral front ends); FP32 matches PyTorch (max diff 3e-4), INT8 dynamic quantization is smaller but distorts the logits and is slower, so it is not used. Decisions:
- **Protocol:** every Phase 3 model trains under the Phase 2 result: silence-trimmed audio and the selected augmentation (E4a: RawBoost + gain), 3 seeds, the same scoring sets as Phase 2 (dev, eval, In-the-Wild, DEEP-VOICE, degraded dev and eval subsets). E4a is the reference row.
- **Models:** `src/adfd/models/lcnn.py` (E5a, 270k params), `rawnet2.py` (E5b, 17.6M), `aasist.py` (E5c, 297,705 params, the paper's size; also accepts SSL feature maps), `crnn.py` (E6a SE-residual CNN + time-major BiLSTM; E6b + attentive statistics pooling; E6c light CNN + BiGRU + ASP), `src/adfd/features/ssl.py` (WavLM / wav2vec 2.0 / XLS-R front end, learned layer weights, fine-tuned or frozen).
- **AASIST on 4 GB:** batch 12 × 2 accumulation (effective 24, as in the paper) and a 20-epoch cosine schedule instead of 100 epochs (about 20 min per epoch). The paper-reproduction run (untrimmed, no augmentation) was skipped by choice; the port matches the reference parameter count exactly.
- **P3.3 cut to an 8 h budget (user decision):** one SSL model, **E7a = WavLM-Base+ (fine-tuned, CNN encoder frozen, gradient checkpointing) + the E6b back end (SSL-CNN-BiLSTM-Att)**, 3 seeds, 3 epochs, batch 8 × 4, SSL LR 1e-5, back-end LR 1e-4; peak VRAM 3.2 GB. Not run: wav2vec2-base, XLS-R-300M (frozen) and WavLM + AASIST; the code supports all three (`frontend: {name: ssl, model: ..., mode: frozen}`, `aasist` with `input: features`). Large models would run frozen on the fly rather than from a feature cache, so augmentation still applies.
- **P3.4:** `scripts/fuse_scores.py`: per-seed logistic regression on dev scores of the best spectrogram model (by degraded-dev EER), AASIST and E7a, plus every pair.
- **P3.5:** `scripts/efficiency.py` (parameters, checkpoint size, GPU and CPU speed); `scripts/make_tables_p3.py` builds the main table and picks the app model on dev data only (lowest degraded-dev EER). ONNX export follows once the app model is known (needs the `onnx` and `onnxruntime` packages).

---

## Phase 4: Reliability and explainability

- [x] **P4.1 Calibration:** fit temperature or Platt scaling on **dev** scores. Report Expected Calibration Error before and after, with reliability diagrams on all three test sets. The app then shows real probabilities.
- [x] **P4.2 Three-way decision:** choose two thresholds on dev, giving **Real / Uncertain / Fake**.
- [x] **P4.3 Out-of-distribution check:** flag non-speech or unusual input (music, tones, silence, noise) using voice-activity speech ratio, an energy/entropy check, and the Mahalanobis distance of the model's internal features from the training data. Test it on the tone and noise examples that currently score "fake" at 0.99.
- [x] **P4.4 Segment timeline:** sliding 2–4 s windows, giving a score-over-time curve and the most suspicious segments. Test it on DEEP-VOICE: clips with a converted segment spliced into real speech should show a visible spike.
- [ ] **P4.5 Grad-CAM / Integrated Gradients** *(prototype: gradient × input saliency on the input features, in the app and explain.py; Grad-CAM, attention plots and the per-band analysis are open)*: heatmaps over the time-frequency input for spectrogram models, and attention-weight plots for the pretrained-feature models. Analyse which frequency bands matter for each dataset and attack.
- [ ] **P4.6 Error analysis:** listen to and summarise the worst false accepts (fakes called real) and false rejects (real called fake) per test set. This feeds directly into the paper's discussion section.

**Done when:** `explain.py` produces a timeline, a heatmap and a calibrated verdict for any audio file, and the calibration and out-of-distribution results are in `experiments/tables/`.

**Status (2026-11-21): prototype done** (P4.1–P4.4; P4.5 partly; P4.6 open).
- `src/adfd/calibration.py`: Platt scaling with balanced classes, fitted on clean dev scores plus the 10 degraded dev conditions when available (clean dev alone is almost separable and gives over-confident probabilities); ECE and reliability diagrams; Real / Uncertain / Fake thresholds with at most 5% of each dev class on the wrong side.
- `src/adfd/ood.py`: one-sided signal checks (level, silence ratio, spectral flatness, tonality, voicing, syllabic rhythm) with limits from bona fide dev speech, plus a Mahalanobis check on the penultimate-layer embedding. E4a: flags all 8 synthetic non-speech inputs (tones that score P(fake) = 1.00, noise, hum, chord, clicks, silence) and 2.5–7.5% of real or fake test speech.
- `src/adfd/inference.py` (`Analyzer`): whole-file analysis with 4 s windows, 1 s hop, mean logit, timeline, suspicious segments, saliency. `python -m adfd.explain <file>` writes verdict.json, timeline.png and heatmap.png.
- `scripts/fit_reliability.py --run runs/<run>` writes `<run>/reliability.json` and `experiments/reliability/<run>.{md,json,png}`. E4a ECE: ASVspoof eval 22.4 → 14.7%, In-the-Wild 57.7 → 31.7%. Fitted for E1, E2a, E4a, E4c and E5a so far.

---

## Phase 5: App feature upgrade

Rebuild the Streamlit app on the PyTorch model. With TensorFlow gone, upgrade Streamlit and remove the `starlette` pin.

| Feature | Description | Priority |
|---|---|---|
| Calibrated three-way verdict | Real / Uncertain / Fake with a real probability | Must |
| Out-of-distribution warning | "This doesn't look like speech", instead of a confident wrong answer | Must |
| Score timeline | Score over time, with suspicious segments highlighted and playable | Must |
| Spectrogram and heatmap | The input spectrogram with the Grad-CAM overlay | Should |
| Model selector and ensemble view | Choose a model, or show every model's score side by side and whether they agree | Should |
| Batch mode | Upload several files or a ZIP, get a CSV of results | Should |
| PDF/HTML report | Verdict, timeline, heatmap, model and version info, limitations | Should |
| Microphone mode | Record in the browser, rolling verdict | Could |
| Video input | Extract the audio track with ffmpeg and analyse it | Could |
| Demo samples | Built-in examples from DEEP-VOICE `DEMONSTRATION` and public-domain clips | Could |

**Done when:** every "Must" and "Should" feature works in the running app with the real model, and the server terminal shows no warnings (checked by the same kind of upload test used for the current app).

**Status (2026-11-21): prototype done.** `app.py` (Streamlit 1.67, PyTorch, CPU inference) has every Must and Should feature: calibrated three-way verdict, OOD warning, timeline with playable suspicious segments, spectrogram + saliency heatmap, model selector and an all-models comparison with an agreement count, batch mode (files or ZIP → CSV), HTML report; plus demo samples (DEEP-VOICE `DEMONSTRATION`, read from the dataset folder, not shipped). Checked in the browser on both demonstration clips; the server log shows no warnings. Open: microphone mode and video input (Could). The v1 TensorFlow app is in `legacy/app_v1_tensorflow.py`.

**Merged detector (2026-11-22).** Two external CNN-BiLSTM log-mel checkpoints (VoiceGuard improved, VoiceGuard v2; `src/adfd/models/voiceguard.py`) were merged with E7a into `src/adfd/ensemble.py`, now the app/API default. Scores: `scripts/score_voiceguard.py`; fusion and decision rule: `scripts/fit_ensemble.py` → `experiments/ensemble/ensemble.md` (tuned on ASVspoof dev + degraded dev only). **VoiceGuard v2 was trained on ~79% of In-the-Wild**, so In-the-Wild is no longer a clean test for it or for the merged detector. The VoiceGuard models are near chance on ASVspoof and get ≈ 0 fusion weight; they act through an agreement gate (Fake needs ≥ 2 of 3 models) and a veto (no Real while any model has P(fake) ≥ 0.95). This cuts real → Fake from 0.11 / 6.74 / 4.67% (E7a alone, at 0.5) to 0.03 / 1.72 / 0.00% (ASVspoof eval / In-the-Wild / DEEP-VOICE), at the cost of more Uncertain answers. Voice conversion (DEEP-VOICE, RVC) remains mostly missed: about half of fake windows are still called Real. Explanations: `src/adfd/forensics.py` (Real/Uncertain/Fake regions over time, mel-spectrogram saliency per band, model-free acoustic measurements against 1,700 real reference clips, `scripts/fit_forensics.py`). Microphone and video input (PyAV) added to the app.

---

## Phase 6: Deployment and missing basics

- [x] **FastAPI service** (`api/`): `POST /predict` (file → verdict, probability, timeline) and `GET /health`, with an OpenAPI description.
- [ ] **Docker** *(Dockerfile, Dockerfile.gpu and docker-compose.yml written; not built here because Docker is not installed)*: separate CPU and GPU images, `docker compose up` runs the app and API together.
- [ ] **Tests and CI** *(pytest suite and `.github/workflows/ci.yml` written; CI not run until pushed)*: pytest for manifests, feature shapes, metric correctness (EER on synthetic scores), model loading, and API and app smoke tests. A GitHub Actions workflow runs the CPU tests.
- [x] **Model card** (`docs/MODEL_CARD.md`): training data, intended use, results per dataset, known failure cases, and limits on misuse.
- [x] **README rewrite:** honest results table, setup instructions, dataset instructions (links and expected folder layout, **no redistributed data**), architecture diagram.
- [ ] **Model weights:** remove the three duplicate `.h5`/`.keras` files from the main tree. Publish weights as GitHub Release assets or on the Hugging Face Hub, with download-on-first-run in the app.
- [ ] **LICENSE and CITATION.cff.** Note the licence terms of each dataset.
- [ ] **Hugging Face Space** for a public demo (CPU build of the ONNX model).
- [ ] Move the original notebooks to `legacy/` with a note that their reported numbers are superseded.

**Done when:** a fresh clone, plus `docker compose up` or `pip install` with the documented steps, runs the app and API, and CI is green.

---

## Phase 7: Paper and release

- [ ] **Tables:**
  1. Main results: every model × 3 test sets (EER, plus min t-DCF on ASVspoof), mean ± std over seeds.
  2. Feature ablation.
  3. Architecture and pretrained-front-end ablation.
  4. Augmentation ablation.
  5. Silence-shortcut check.
  6. Calibration.
  7. Efficiency.
- [ ] **Figures:** DET curves, per-attack EER (A07–A19), robustness curves, a timeline example, Grad-CAM examples, reliability diagrams.
- [ ] **Paper structure:** Introduction → Related work → Datasets and protocol (stress the leak-free splits) → Method (SSL-CNN-BiLSTM-Att) → Experiments → Explainability → Limitations and ethics → Conclusion.
- [ ] Tag a release (`v2.0`) with weights, results CSVs and the exact configs used for every table.

---

## Experiment matrix (all runs)

| ID | Model | Input | Augmentation | Seeds | Purpose |
|---|---|---|---|---|---|
| E0 | CNN-BiLSTM (legacy setup) | MFCC | none | 1 | Show how inflated the old result was |
| E1 | CNN-BiLSTM | MFCC | none | 3 | Honest baseline |
| E2a–d | CNN-BiLSTM | LFCC / CQCC / log-mel / raw | none | 1 (3 for best) | Feature ablation |
| E3 | CNN-BiLSTM | best of E2, silence trimmed vs not | none | 1 | Shortcut check |
| E4a–c | CNN-BiLSTM | best of E2 | RawBoost / codec / both | 1 (3 for best) | Augmentation ablation |
| E5a–c | LCNN / RawNet2 / AASIST | LFCC / raw / raw | best of E4 | 3 | Published baselines |
| E6a–b | CNN-BiLSTM + SE + attentive pooling | best of E2 | best of E4 | 3 | Architecture upgrade |
| E7a–c | **SSL-CNN-BiLSTM-Att** (WavLM-Base+ / wav2vec2-base / XLS-R frozen) | raw → pretrained model | best of E4 | 3 | **Main contribution** |
| E8 | WavLM + AASIST | raw → pretrained model | best of E4 | 3 | Strong reference |
| E9 | Fusion of best 3 | scores | — | — | Ensemble |

Each run is scored on: ASVspoof eval (overall and per attack), In-the-Wild and DEEP-VOICE, plus the degraded ASVspoof eval sets from P2.4 for the headline models.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| 4 GB VRAM too small for pretrained models | Base-size models, mixed precision, gradient checkpointing, frozen and cached Large models, LoRA, 8-bit optimisers |
| Slow training on a laptop GPU | Short 4 s crops, cached features for non-pretrained runs, small runs before full ones; use Kaggle or Colab GPUs for the heaviest runs (the same code runs there) |
| C: drive fills up | All caches and outputs on G: (environment variables set in P0.B) |
| Test sets leak into training or tuning | Test sets are only ever scored; thresholds and calibration use dev only; manifest checks in CI |
| DEEP-VOICE is small and imbalanced (8 real vs 56 fake recordings) | Report per-window and per-recording EER, bootstrap confidence intervals, and frame it as a case study |
| Cross-dataset results stay poor | Still publishable: the gap, the shortcut analysis and the ablations are themselves the contribution, so report them honestly |
| Dataset licence limits | Never redistribute audio; document how to download each dataset |

---

## First concrete steps

1. P0.B: create the PyTorch environment with caches on G:, and confirm `torch.cuda.is_available()`.
2. P0.A and P0.C: restructure the repo and build and validate the 3 manifests.
3. P1.1–P1.5: port the CNN-BiLSTM and the evaluation harness, train E1, and record the honest baseline on all 3 test sets.
4. Review the baseline numbers together, then start the P2 ablations.
