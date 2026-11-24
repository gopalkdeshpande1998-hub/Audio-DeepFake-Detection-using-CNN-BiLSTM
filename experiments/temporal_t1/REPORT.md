# T1 segment-level detector: results (2026-11-24)

> **Update (2026-11-24, later):** the app no longer has an UNCERTAIN class. Silent parts are NO SPEECH (grey,
> not classified); every speech segment is REAL or FAKE by argmax of its calibrated probabilities (FAKE only if P(fake) > P(real), ties REAL); confidence is
> informational. The model and its calibration are unchanged. The frame metrics below were measured at the
> validation EER threshold 0.49; at the app's 0.5 the threshold sweep gives, on the 6,000 test utterances,
> precision 0.938, recall 0.874, F1 0.905, false-positive rate 5.7% (vs 0.936 / 0.876 / 0.905 / 5.9% at 0.49).
> The "UNCERTAIN band" row and the example's UNCERTAIN / NO SPEECH regions describe the earlier version.

Run `runs/t1_wavlm_crnn_temporal_s42` (config `configs/t1_wavlm_crnn_temporal.yaml`), 1 seed, 2 epochs
(37 min on the RTX 3050). Raw numbers: `test.json`, `test_long.json`; calibration:
`runs/t1_wavlm_crnn_temporal_s42/temporal_calibration.json`.

## What was built

- **Model**: E7a (WavLM-Base+ -> SE-res CNN -> BiLSTM -> attentive pooling -> clip head) plus a frame head on the
  BiLSTM output, before any pooling: one logit per 0.16 s. Warm start from E7a; the clip head is kept.
- **Training data**: PartialSpoof v1.2 dev, 16 speakers (21,920 utterances, 87% partially fake, 0.16-s labels)
  + 5,160 fully real / fully fake ASVspoof 2019 LA train clips. Loss: class-weighted frame BCE + 0.5 x clip BCE.
- **Validation / calibration**: 4 held-out dev speakers (2,924 utterances): Platt scaling of frame logits, decision
  threshold 0.49 (validation EER), UNCERTAIN band 0.29-0.69 (98% validation accuracy outside it, at least +-0.15).
- **Test**: PartialSpoof v1.2 eval only (67 speakers and attacks A07-A19 never seen in training): a seeded,
  kind-stratified sample of 6,000 utterances (131,212 frames), and 300 recordings of 10-15 s built by joining real
  and partially fake eval utterances of one speaker (labels concatenated exactly, seeded).
- **Baseline**: the old app timeline: E7a whole-clip score on 4-s windows every 0.5 s, read at each 0.16-s frame,
  FAKE at 50%.

## Results on the test set (frames of 0.16 s; FAKE = positive class)

| | T1 (new) 6,000 utts | E7a sliding (old) | T1 (new) 10-15 s mixes | E7a sliding (old) |
|---|---|---|---|---|
| Frame EER | **9.3%** | 32.5% | **10.2%** | 28.9% |
| Frame accuracy | 90.9% | 67.2% | 90.7% | 69.7% |
| Frame precision / recall | 0.936 / 0.876 | 0.654 / 0.720 | 0.896 / 0.849 | 0.572 / 0.755 |
| Frame F1 | **0.905** | 0.686 | **0.872** | 0.651 |
| False-positive rate (real frames called fake) | 5.9% | 37.5% | 5.9% | 33.8% |
| False-negative rate (fake frames missed) | 12.4% | 28.0% | 15.1% | 24.5% |
| Segment precision / recall (IoU >= 0.5) | 0.768 / 0.824 | 0.474 / 0.090 | 0.716 / 0.781 | 0.280 / 0.045 |
| Segment F1 | **0.795** | 0.151 | **0.747** | 0.077 |
| Boundary error, matched regions (mean start / end) | 0.03 / 0.03 s | 0.67 / 0.20 s | 0.06 / 0.06 s | 0.65 / 0.68 s |
| Fake regions < 0.5 s detected | **91.2%** (n=11,344) | 46.5% | **88.7%** (n=1,279) | 48.9% |
| 0.5-1 s / 1-2 s / >= 2 s detected | 91.6 / 88.2 / 92.2% | 60.2 / 79.4 / 92.3% | 91.9 / 86.6 / 91.0% | 61.4 / 81.5 / 95.2% |
| With the UNCERTAIN band: share uncertain / accuracy of the rest | 7.3% / 93.4% | 4.3% / 67.9% | 7.3% / 93.3% | 7.2% / 71.2% |
| Clip accuracy (any fake in the clip) | 96.2% (clip head), 97.9% (any FAKE frame) | 59.6% | 89.3% (clip head), 99.7% (any FAKE frame) | 53.7% |

Confusion matrix, T1, 6,000 test utterances (frames): fake called fake 57,066; real called fake 3,871; fake called
real 8,085; real called real 62,190. Clip level: 4 of 619 fully real utterances called fake (0.6%), 222 of 5,381
utterances with fake parts missed (4.1%).

Threshold analysis (T1 frames, 6,000 utterances; precision / recall / F1 / FPR):
0.1: 0.846 / 0.944 / 0.892 / 17.0% · 0.3: 0.906 / 0.908 / 0.907 / 9.3% · 0.5: 0.938 / 0.874 / 0.905 / 5.7% ·
0.7: 0.963 / 0.823 / 0.887 / 3.2% · 0.9: 0.982 / 0.731 / 0.838 / 1.3%. F1 is flat between 0.2 and 0.6, so the
validation threshold (0.49) is not finely tuned to this test set.

Validation (held-out dev speakers, same attacks as training): frame EER 3.2%, clip EER 1.0%. The test frame EER
(9.3%) is about three times higher: unseen attacks are harder.

## Example (browser check)

12-s recording from eval speaker LA_0001, truth REAL 0-3.04 s, FAKE 3.04-6.56 s, REAL 6.56-9.12 s, FAKE 9.12-11.68 s.
T1: NO SPEECH 0-0.8, REAL 0.8-2.9 (one 0.2-s UNCERTAIN), FAKE 3.0-6.6 (99%), NO SPEECH 6.6-7.7, REAL 7.7-8.6,
UNCERTAIN 8.6-9.0, FAKE 9.4-11.4 (100%). Clicking the FAKE block plays 3.0 s onward with the row highlighted.

## Limitations (read before relying on it)

- **One seed, two epochs.** No variance estimate yet; a longer run or more seeds may change the numbers.
- **One corpus family.** Training and test both come from PartialSpoof / ASVspoof 2019: English, studio quality,
  2019-era TTS and VC. Test attacks are unseen, but the splicing procedure is the same, so the model may partly rely
  on PartialSpoof's concatenation artefacts. Not yet measured: real-world partially edited audio, modern voice
  conversion (DEEP-VOICE), codecs and noise.
- **Boundary error** is measured on matched regions only (IoU >= 0.5), at 0.16-s resolution; it says nothing about
  regions the model missed.
- **"Detected" short regions** means at least half of the region's frames were called FAKE; PartialSpoof's 0.16-s
  labels mark a frame fake if any part of it is fake.
- **Silence** is labelled NO SPEECH by an energy rule in the app; the metrics above score every labelled frame,
  including silence, exactly as PartialSpoof labels it.
- The whole-clip verdict in the app is still E7a (unchanged); the 10-15 s mixes all contain fake parts, so their
  clip accuracy measures misses only.
