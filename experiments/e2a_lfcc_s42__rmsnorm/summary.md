# Results: e2a_lfcc_s42__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 28.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.15 |  | 0.0022 | 1.0000 | 99.84 | 99.99 | 99.84 |
| asvspoof2019_la:eval | 71,237 | 7.45 |  | 0.2027 | 0.9700 | 74.94 | 99.84 | 72.17 |
| in_the_wild:test | 31,779 | 37.97 |  |  | 0.6282 | 37.30 | 36.90 | 96.63 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.35 | 0.14 | 0.08 | 3.82 | 2.20 | 2.85 | 3.24 | 0.77 | 2.48 | 0.98 | 31.46 | 15.28 | 1.12 |
