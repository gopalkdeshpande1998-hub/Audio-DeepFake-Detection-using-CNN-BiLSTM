# Results: e2a_lfcc_s43__trimmed

Model: cnn_bilstm (786,625 parameters), best dev epoch 29.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.92 | 1.65–2.17 | 0.0558 | 0.9977 | 98.95 | 99.23 | 99.61 |
| asvspoof2019_la:eval | 71,237 | 21.78 | 21.36–22.24 | 0.4773 | 0.8819 | 66.17 | 99.14 | 62.82 |
| in_the_wild:test | 31,779 | 42.02 | 41.44–42.61 |  | 0.6030 | 37.37 | 36.89 | 96.30 |
| deep_voice:test | 14,896 | 46.03 | 32.51–54.87 |  | 0.5774 | 88.31 | 88.43 | 99.69 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.6406 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.38 | 0.28 | 0.38 | 25.40 | 16.40 | 38.10 | 29.30 | 3.52 | 17.25 | 0.53 | 26.55 | 49.69 | 0.59 |
