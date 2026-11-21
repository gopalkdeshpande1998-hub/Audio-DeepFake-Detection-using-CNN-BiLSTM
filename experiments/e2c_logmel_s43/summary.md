# Results: e2c_logmel_s43

Model: cnn_bilstm (786,625 parameters), best dev epoch 5.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.08 | 0.04–0.20 | 0.0025 | 1.0000 | 99.92 | 99.99 | 99.91 |
| asvspoof2019_la:eval | 71,237 | 6.99 | 6.82–7.18 | 0.1078 | 0.9680 | 84.22 | 99.98 | 82.42 |
| in_the_wild:test | 31,779 | 74.07 | 73.55–74.57 |  | 0.2120 | 33.62 | 34.50 | 87.43 |
| deep_voice:test | 14,896 | 48.71 | 30.00–61.79 |  | 0.5465 | 89.02 | 88.89 | 99.94 |
| deep_voice:test (per recording) | 64 | 50.00 |  |  | 0.5201 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 1.61 | 0.04 | 0.75 | 0.55 | 0.37 | 0.34 | 0.73 | 1.28 | 0.07 | 43.28 | 2.40 | 0.12 |
