# Merged detector: E7a + VoiceGuard improved + VoiceGuard v2

All numbers below are produced by `scripts/fit_ensemble.py`. Tuning data: ASVspoof 2019 LA dev (clean + 10 degraded conditions). Test sets never set any weight or threshold.

**Contamination warning:** VoiceGuard v2 was trained on about 79% of In-the-Wild (speaker split; the split files are not available). Its In-the-Wild numbers, and those of any fusion that includes it, are **not test results**. ASVspoof eval and DEEP-VOICE are clean for all three members (as far as is known: the training data of VoiceGuard improved is not recorded).

## EER (%) of each member and fusion

| System | tuning (dev + degraded dev) | ASVspoof eval | In-the-Wild | DEEP-VOICE (windows) | DEEP-VOICE (recording) |
|---|---|---|---|---|---|
| e7a_wavlm_crnn_s42 | 3.76 | 6.22 | 10.67 | 23.68 | – |
| vg_improved | 49.77 | 47.04 | 36.15 | 30.39 | – |
| vg_v2 | 44.49 | 36.25 | 3.83 | 36.19 | – |
| fusion: e7a (w = 1.00, 0.00, 0.00) | 3.76 | 6.22 | 10.67 | 23.68 | 12.50 |
| fusion: equal (w = 0.33, 0.33, 0.33) | 26.79 | 25.98 | 3.67 | 20.84 | 12.50 |
| fusion: lr (w = 1.03, -0.05, -0.06) | 3.67 | 8.27 | 12.80 | 26.37 | 24.11 |

## Decision rule (fusion = lr)

Fake if fused P(fake) ≥ 0.763 and at least 2 members vote fake; Real if fused P(fake) ≤ 0.174 and no member has P(fake) ≥ 0.95; otherwise Uncertain (thresholds: ≤ 2.0% of each tuning class on the wrong side).

| Rule | Set | decided % | error among decided % | real → Fake % (false detection) | fake → Real % | fakes caught % | reals confirmed % |
|---|---|---|---|---|---|---|---|
| fused thresholds only | tuning (dev + degraded dev) | 95.7 | 2.09 | 2.00 | 2.00 | 93.8 | 93.4 |
| fused thresholds only | ASVspoof eval | 93.2 | 14.53 | 0.05 | 15.10 | 77.4 | 99.8 |
| fused thresholds only | In-the-Wild | 88.8 | 9.04 | 5.32 | 12.60 | 70.1 | 87.1 |
| fused thresholds only | DEEP-VOICE (windows) | 71.9 | 61.36 | 4.24 | 49.84 | 19.5 | 86.1 |
| + agreement gate and veto (used) | tuning (dev + degraded dev) | 64.0 | 2.75 | 1.84 | 1.73 | 59.1 | 71.9 |
| + agreement gate and veto (used) | ASVspoof eval | 65.2 | 6.69 | 0.03 | 4.86 | 60.7 | 61.8 |
| + agreement gate and veto (used) | In-the-Wild | 81.1 | 2.53 | 1.72 | 2.61 | 67.7 | 85.8 |
| + agreement gate and veto (used) | DEEP-VOICE (windows) | 54.7 | 76.75 | 0.00 | 48.01 | 2.3 | 85.9 |

## For comparison: each member alone, plain 0.5 threshold, no abstention

| Member | Set | error % | real → Fake % | fake → Real % |
|---|---|---|---|---|
| e7a_wavlm_crnn_s42 | ASVspoof eval | 16.65 | 0.11 | 18.56 |
| e7a_wavlm_crnn_s42 | In-the-Wild | 11.14 | 6.74 | 18.58 |
| e7a_wavlm_crnn_s42 | DEEP-VOICE (windows) | 65.47 | 4.67 | 74.16 |
| vg_improved | ASVspoof eval | 37.97 | 62.13 | 35.19 |
| vg_improved | In-the-Wild | 34.88 | 30.01 | 43.13 |
| vg_improved | DEEP-VOICE (windows) | 83.16 | 0.38 | 94.98 |
| vg_v2 | ASVspoof eval | 39.50 | 33.64 | 40.17 |
| vg_v2 | In-the-Wild | 3.18 | 1.73 | 5.62 |
| vg_v2 | DEEP-VOICE (windows) | 83.43 | 0.59 | 95.26 |
