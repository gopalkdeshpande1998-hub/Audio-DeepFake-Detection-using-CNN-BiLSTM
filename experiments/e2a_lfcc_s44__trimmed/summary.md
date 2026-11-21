# Results: e2a_lfcc_s44__trimmed

Model: cnn_bilstm (786,625 parameters), best dev epoch 19.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 3.92 | 3.46–4.25 | 0.1248 | 0.9903 | 97.79 | 98.14 | 99.42 |
| asvspoof2019_la:eval | 71,237 | 22.35 | 21.80–22.77 | 0.5079 | 0.8702 | 71.29 | 97.99 | 69.40 |
| in_the_wild:test | 31,779 | 54.64 | 54.10–55.23 |  | 0.4279 | 37.42 | 36.88 | 96.01 |
| deep_voice:test | 14,896 | 44.03 | 26.47–60.94 |  | 0.5970 | 87.51 | 87.70 | 99.71 |
| deep_voice:test (per recording) | 64 | 48.21 |  |  | 0.5670 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.68 | 0.59 | 1.02 | 24.93 | 17.48 | 38.56 | 34.80 | 6.13 | 15.24 | 1.44 | 28.14 | 50.69 | 1.40 |
