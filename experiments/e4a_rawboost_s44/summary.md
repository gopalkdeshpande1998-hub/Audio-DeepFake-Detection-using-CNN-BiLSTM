# Results: e4a_rawboost_s44

Model: cnn_bilstm (786,625 parameters), best dev epoch 40.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.16 | 0.08–0.32 | 0.0048 | 1.0000 | 99.84 | 99.98 | 99.84 |
| asvspoof2019_la:eval | 71,237 | 13.92 | 13.51–14.33 | 0.3404 | 0.9406 | 59.16 | 99.98 | 54.47 |
| in_the_wild:test | 31,779 | 58.83 | 58.23–59.36 |  | 0.4109 | 38.51 | 35.43 | 79.46 |
| deep_voice:test | 14,896 | 48.28 | 26.30–61.89 |  | 0.5562 | 89.04 | 89.49 | 99.11 |
| deep_voice:test (per recording) | 64 | 63.39 |  |  | 0.4152 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.24 | 0.02 | 0.02 | 29.77 | 6.65 | 39.01 | 16.10 | 0.20 | 6.46 | 0.27 | 10.07 | 13.11 | 0.12 |
