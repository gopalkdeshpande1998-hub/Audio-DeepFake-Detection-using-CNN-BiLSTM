# Results: e4a_rawboost_s43

Model: cnn_bilstm (786,625 parameters), best dev epoch 40.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.31 | 0.23–0.47 | 0.0101 | 0.9996 | 99.68 | 99.96 | 99.68 |
| asvspoof2019_la:eval | 71,237 | 13.60 | 13.27–13.95 | 0.3218 | 0.9420 | 60.88 | 99.98 | 56.39 |
| in_the_wild:test | 31,779 | 47.35 | 46.83–47.88 |  | 0.5426 | 38.75 | 35.64 | 80.37 |
| deep_voice:test | 14,896 | 35.66 | 18.20–55.10 |  | 0.7050 | 88.82 | 89.54 | 98.76 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.7589 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.53 | 0.02 | 0.14 | 28.59 | 7.67 | 38.68 | 17.93 | 0.18 | 5.00 | 0.31 | 8.30 | 12.63 | 0.18 |
