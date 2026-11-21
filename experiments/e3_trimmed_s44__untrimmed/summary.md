# Results: e3_trimmed_s44__untrimmed

Model: cnn_bilstm (786,625 parameters), best dev epoch 29.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.04 | 0.01–0.07 | 0.0004 | 1.0000 | 99.88 | 100.00 | 99.87 |
| asvspoof2019_la:eval | 71,237 | 22.01 | 21.70–22.38 | 0.3455 | 0.8471 | 63.42 | 100.00 | 59.21 |
| in_the_wild:test | 31,779 | 51.21 | 50.57–51.77 |  | 0.4913 | 36.73 | 36.25 | 92.46 |
| deep_voice:test | 14,896 | 36.68 | 21.68–53.22 |  | 0.6778 | 86.82 | 87.78 | 98.67 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.7143 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.06 | 0.00 | 0.00 | 47.76 | 1.71 | 52.03 | 52.36 | 0.08 | 3.19 | 0.20 | 10.81 | 44.24 | 0.04 |
