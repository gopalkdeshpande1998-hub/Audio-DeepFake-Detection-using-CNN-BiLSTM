# Merged detector, two-way REAL / FAKE decision

Produced by `scripts/fit_ensemble_binary.py`. Weights and calibration come from the tuning sets only (ASVspoof dev + degraded dev, DEEP-VOICE non-test persons). Decision: FAKE if merged P(fake) >= 0.5.

Weights (E7a, VoiceGuard improved, VoiceGuard v2): 0.45, 0.35, 0.20 (each constrained to 0.2–0.45); member log-odds clipped to ±4.0; calibration P = sigmoid(0.601·s + 0.120), i.e. FAKE when s ≥ -0.20. Constraint: two members that agree at full confidence always beat the third (margin 0.2).

**In-the-Wild is not a clean test**: VoiceGuard v2 was trained on most of it.

## Accuracy at the 0.5 decision (%)

| System | Set | n | accuracy | balanced acc. | real → FAKE | fake → REAL | EER |
|---|---|---|---|---|---|---|---|
| merged (chosen) | ASVspoof dev + degraded dev | 91,044 | 82.0 | 85.1 | 8.8 | 21.1 | 10.61 |
| merged (chosen) | DEEP-VOICE tuning persons | 11,172 | 16.2 | 52.8 | 0.0 | 94.4 | 26.02 |
| merged (chosen) | DEEP-VOICE tuning persons (per recording) | 48 | 14.6 | 51.2 | 0.0 | 97.6 | 16.67 |
| merged (chosen) | ASVspoof eval | 71,237 | 81.4 | 84.7 | 11.2 | 19.4 | 14.26 |
| merged (chosen) | DEEP-VOICE test persons | 3,724 | 19.9 | 52.3 | 0.0 | 95.5 | 21.24 |
| merged (chosen) | DEEP-VOICE test persons (per recording) | 16 | 12.5 | 50.0 | 0.0 | 100.0 | 0.00 |
| merged (chosen) | In-the-Wild (VG v2 contaminated) | 31,779 | 92.9 | 91.5 | 2.9 | 14.0 | 6.62 |
| E7a alone (for comparison) | ASVspoof dev + degraded dev | 91,044 | 95.9 | 96.2 | 3.2 | 4.3 | 3.76 |
| E7a alone (for comparison) | DEEP-VOICE tuning persons | 11,172 | 32.9 | 59.7 | 5.6 | 74.9 | 26.64 |
| E7a alone (for comparison) | DEEP-VOICE tuning persons (per recording) | 48 | 27.1 | 58.3 | 0.0 | 83.3 | 16.67 |
| E7a alone (for comparison) | ASVspoof eval | 71,237 | 83.3 | 90.7 | 0.1 | 18.6 | 6.22 |
| E7a alone (for comparison) | DEEP-VOICE test persons | 3,724 | 39.3 | 62.8 | 2.7 | 71.8 | 18.72 |
| E7a alone (for comparison) | DEEP-VOICE test persons (per recording) | 16 | 25.0 | 57.1 | 0.0 | 85.7 | 0.00 |
| E7a alone (for comparison) | In-the-Wild (VG v2 contaminated) | 31,779 | 88.9 | 87.3 | 6.7 | 18.6 | 10.67 |
| equal weights (for comparison) | ASVspoof dev + degraded dev | 91,044 | 72.8 | 76.8 | 15.3 | 31.2 | 20.43 |
| equal weights (for comparison) | DEEP-VOICE tuning persons | 11,172 | 16.0 | 52.6 | 0.0 | 94.7 | 26.11 |
| equal weights (for comparison) | DEEP-VOICE tuning persons (per recording) | 48 | 12.5 | 50.0 | 0.0 | 100.0 | 16.67 |
| equal weights (for comparison) | ASVspoof eval | 71,237 | 79.2 | 77.5 | 24.6 | 20.4 | 22.67 |
| equal weights (for comparison) | DEEP-VOICE test persons | 3,724 | 19.7 | 52.2 | 0.0 | 95.7 | 22.74 |
| equal weights (for comparison) | DEEP-VOICE test persons (per recording) | 16 | 12.5 | 50.0 | 0.0 | 100.0 | 0.00 |
| equal weights (for comparison) | In-the-Wild (VG v2 contaminated) | 31,779 | 95.7 | 95.1 | 2.6 | 7.2 | 4.27 |

## EER (%) of each member alone

| Member | ASVspoof dev + degraded dev | DEEP-VOICE tuning persons | ASVspoof eval | DEEP-VOICE test persons | In-the-Wild (VG v2 contaminated) |
|---|---|---|---|---|---|
| e7a_wavlm_crnn_s42 | 3.76 | 26.64 | 6.22 | 18.72 | 10.67 |
| vg_improved | 49.77 | 36.55 | 47.04 | 18.72 | 36.15 |
| vg_v2 | 44.49 | 30.62 | 36.25 | 46.32 | 3.83 |
