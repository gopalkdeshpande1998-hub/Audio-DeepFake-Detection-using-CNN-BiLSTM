# Results: e2a_lfcc_s42

Model: cnn_bilstm (786,625 parameters), best dev epoch 28.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.04 | 0.01–0.08 | 0.0006 | 1.0000 | 99.96 | 100.00 | 99.96 |
| asvspoof2019_la:eval | 71,237 | 5.95 | 5.74–6.17 | 0.1347 | 0.9747 | 74.13 | 99.99 | 71.16 |
| in_the_wild:test | 31,779 | 38.34 | 37.83–38.85 |  | 0.6379 | 37.95 | 37.23 | 97.50 |
| deep_voice:test | 14,896 | 49.25 | 34.34–59.58 |  | 0.5202 | 87.22 | 87.61 | 99.47 |
| deep_voice:test (per recording) | 64 | 50.89 |  |  | 0.4598 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.04 | 0.02 | 2.12 | 1.12 | 1.73 | 1.56 | 0.20 | 1.29 | 0.14 | 34.71 | 9.34 | 0.10 |
