# Results: e4a_rawboost_s42

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.55 | 0.43–0.71 | 0.0153 | 0.9995 | 99.45 | 99.94 | 99.45 |
| asvspoof2019_la:eval | 71,237 | 15.62 | 15.24–16.05 | 0.3703 | 0.9262 | 60.67 | 99.96 | 56.17 |
| in_the_wild:test | 31,779 | 35.98 | 35.38–36.50 |  | 0.6890 | 41.74 | 38.59 | 95.88 |
| deep_voice:test | 14,896 | 45.06 | 25.58–59.95 |  | 0.5911 | 87.32 | 89.70 | 96.59 |
| deep_voice:test (per recording) | 64 | 50.00 |  |  | 0.5446 | 90.62 | 90.32 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.28 | 0.02 | 0.08 | 31.99 | 12.74 | 40.26 | 18.26 | 0.42 | 13.21 | 0.33 | 16.28 | 15.16 | 0.10 |
