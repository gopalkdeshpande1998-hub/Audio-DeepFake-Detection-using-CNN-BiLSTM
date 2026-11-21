# Results: e3_trimmed_s43__untrimmed

Model: cnn_bilstm (786,625 parameters), best dev epoch 33.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.59 | 0.46–0.70 | 0.0114 | 0.9994 | 98.45 | 100.00 | 98.28 |
| asvspoof2019_la:eval | 71,237 | 22.05 | 21.74–22.43 | 0.3465 | 0.8552 | 64.41 | 99.99 | 60.31 |
| in_the_wild:test | 31,779 | 59.51 | 58.98–60.10 |  | 0.3763 | 35.50 | 36.00 | 94.44 |
| deep_voice:test | 14,896 | 40.86 | 23.69–60.65 |  | 0.6157 | 87.46 | 87.53 | 99.90 |
| deep_voice:test (per recording) | 64 | 50.00 |  |  | 0.5491 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.06 | 0.06 | 0.04 | 55.38 | 0.53 | 52.85 | 51.32 | 0.28 | 9.12 | 0.14 | 9.59 | 34.27 | 0.00 |
