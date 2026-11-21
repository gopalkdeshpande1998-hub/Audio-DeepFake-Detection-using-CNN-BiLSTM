# Results: e4d_full_aug_s42

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 12.32 | 11.73–13.02 | 0.3268 | 0.9526 | 87.68 | 98.42 | 87.67 |
| asvspoof2019_la:eval | 71,237 | 18.23 | 17.88–18.62 | 0.4454 | 0.9050 | 76.64 | 98.45 | 75.13 |
| in_the_wild:test | 31,779 | 39.10 | 38.52–39.62 |  | 0.6529 | 51.36 | 41.99 | 80.70 |
| deep_voice:test | 14,896 | 42.52 | 29.45–55.88 |  | 0.6048 | 75.52 | 89.31 | 81.82 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.7098 | 90.62 | 91.67 | 98.21 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 33.34 | 0.63 | 0.90 | 36.08 | 10.25 | 29.35 | 11.15 | 2.93 | 8.87 | 23.66 | 14.82 | 8.75 | 17.40 |
