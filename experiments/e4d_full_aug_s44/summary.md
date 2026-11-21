# Results: e4d_full_aug_s44

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 11.61 | 10.98–12.11 | 0.2739 | 0.9569 | 88.39 | 98.53 | 88.39 |
| asvspoof2019_la:eval | 71,237 | 17.97 | 17.48–18.43 | 0.4599 | 0.9126 | 74.90 | 98.47 | 73.15 |
| in_the_wild:test | 31,779 | 38.03 | 37.48–38.59 |  | 0.6752 | 50.17 | 41.58 | 84.00 |
| deep_voice:test | 14,896 | 47.09 | 35.99–57.77 |  | 0.5451 | 73.75 | 88.18 | 80.84 |
| deep_voice:test (per recording) | 64 | 50.00 |  |  | 0.5893 | 90.62 | 90.32 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 26.94 | 0.20 | 0.30 | 29.43 | 14.74 | 24.58 | 7.08 | 1.32 | 9.85 | 19.23 | 18.64 | 20.35 | 22.65 |
