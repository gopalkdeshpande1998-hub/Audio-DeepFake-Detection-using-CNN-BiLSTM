# Results: e2a_lfcc_s43

Model: cnn_bilstm (786,625 parameters), best dev epoch 29.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.00 | 0.00–0.02 | 0.0001 | 1.0000 | 99.99 | 100.00 | 99.99 |
| asvspoof2019_la:eval | 71,237 | 5.59 | 5.37–5.78 | 0.142 | 0.9748 | 62.35 | 99.99 | 58.02 |
| in_the_wild:test | 31,779 | 39.03 | 38.38–39.59 |  | 0.6436 | 37.49 | 36.90 | 95.96 |
| deep_voice:test | 14,896 | 46.03 | 32.51–54.87 |  | 0.5774 | 88.31 | 88.43 | 99.69 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.6406 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.02 | 0.02 | 1.24 | 1.22 | 1.30 | 1.44 | 0.31 | 3.17 | 0.10 | 33.54 | 8.18 | 0.06 |
