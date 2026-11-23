# Model card: audio deepfake detectors (v2)

## Model details

- **Default app model: the merged detector** (`src/adfd/ensemble.py`): E7a, WavLM-Base+ (fine-tuned, learned
  layer weights) followed by an SE-ResNet CNN-BiLSTM with attentive statistics pooling (96.2M parameters, weight
  0.45), plus two log-mel CNN-BiLSTM checkpoints, VoiceGuard improved (0.35) and VoiceGuard v2 (0.20). Each
  member's calibrated log-odds are clipped to ±4 and weighted, so no single model decides. Weights and the line
  were tuned on ASVspoof dev (clean + degraded) and the DEEP-VOICE non-test persons only.
  Other selectable models: E7a alone, the v1 MFCC CNN-BiLSTM, LFCC CNN-BiLSTMs (Phase 2 pick E4a), LCNN, RawNet2,
  and the SE-ResNet CNN-BiLSTM-Att on LFCC.
- **Output:** a calibrated P(fake) and a two-way REAL / FAKE verdict at 50%, a per-second timeline from 2 s windows,
  flagged segments with evidence, and a **trust lookup**: the held-out accuracy of verdicts with the same confidence
  band and member agreement (`scripts/error_analysis.py`, `src/adfd/trust.py`).
- **Framework:** PyTorch 2.11. Code: `src/adfd/`. Version 0.1.0 (prototype). Runs on CPU at about 16× real time (E7a).

## Intended use

- Research on audio deepfake detection, and as a **screening aid** that tells a person where to look
  (timeline, suspicious segments, heatmap) in English speech recordings.
- **Not** for deciding on its own whether a recording is authentic in legal, journalistic,
  financial or personal-safety matters. Not for surveillance or for judging individuals.

## Training data

ASVspoof 2019 Logical Access **train** (2,580 real and 22,800 fake utterances, 20 speakers, attacks A01–A06,
English, studio quality). Model choice, thresholds and calibration use ASVspoof 2019 LA **dev** only
(plus dev clips degraded with codecs and noise). No test data is used for any decision.

## Evaluation data

- ASVspoof 2019 LA **eval**: 71,237 utterances, unseen attacks A07–A19.
- **In-the-Wild**: 31,779 real and fake recordings of celebrities and politicians from the internet.
- **DEEP-VOICE**: 64 long recordings (8 real, 56 voice-converted); scored per 4 s window and per recording.
- Degraded ASVspoof eval subsets: MP3 and Opus at two bit rates, telephone band, white noise 20/10/5 dB, pink noise.

## Results (EER %; mean ± std over 3 seeds, or 1 seed where marked)

| Model | ASVspoof eval | In-the-Wild | DEEP-VOICE (recording) |
|---|---|---|---|
| v1 CNN-BiLSTM, MFCC | 7.97 ± 0.04 | 47.3 ± 4.6 | 52.4 ± 6.6 |
| E4a CNN-BiLSTM, LFCC + RawBoost | 14.38 ± 1.09 | 47.4 ± 11.4 | 50.3 ± 13.0 |
| E7a WavLM-Base+ + CNN-BiLSTM-Att (default, 1 seed) | 6.22 | 10.67 | 12.50 |

Full tables, per-attack and per-condition results: `experiments/tables/`.
Calibration (expected calibration error before → after) for E7a: ASVspoof eval 14.0 → 12.2%,
In-the-Wild 7.3 → 5.0%; for E4a: ASVspoof eval 22.4 → 14.7%, In-the-Wild 57.7 → 31.7% (`experiments/reliability/`).
For E7a the dev classes barely overlap, so the Real/Fake thresholds meet at p = 0.415 and the
"Uncertain" band is empty: every file gets a Real or Fake verdict.

## Known failure cases

- **Merged detector** (`experiments/error_analysis/error_analysis.md`): on ASVspoof eval it calls 11.2% of real
  clips FAKE and 19.4% of fakes REAL; attacks A10 (71% missed) and A18 (52%) are the hardest. It misses about 95%
  of DEEP-VOICE voice-conversion windows. Verdicts within 10 points of the 50% line are right only 48% of the time
  on ASVspoof eval; the app says so in its trust panel.

- **Real-world fakes are missed more than real speech is wrongly flagged.** With its dev-fitted threshold,
  E7a calls 17% of In-the-Wild fakes and 71% of DEEP-VOICE fake windows "Real"; it calls the DEEP-VOICE
  demonstration fake (voice conversion) Real and the real demonstration clip Real. Models trained from scratch
  (E4a and others) are near chance on In-the-Wild. The training data has 20 studio speakers and 2019-era attacks.
- **Short clips** (< 4 s) are repeated to fill the window, as in training; very short clips (< 1 s) are flagged.
- **Non-speech** (tones, noise, music, silence) produces confident but meaningless scores. The
  out-of-distribution check flags 6 of 8 synthetic test signals for E7a (pink noise and a music-like chord pass;
  all 8 for E4a). It also flags 0–10% of real test speech and up to 27% of In-the-Wild fakes for E7a.
- **Compression and noise** raise the error, by about 10 EER points on average for E4a (more for unaugmented models).
- Languages other than English, children's voices, singing, and recent voice-cloning systems are untested.

## Ethical considerations and misuse

A detector can be used to discredit genuine recordings ("it's a deepfake") or to tune a generator
until it passes. Report verdicts with their uncertainty, keep a human in the loop, and never present
a single score as proof. The datasets are used under their own licences and are not redistributed.
