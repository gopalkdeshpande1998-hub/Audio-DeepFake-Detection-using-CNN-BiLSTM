# Results: e2b_cqcc_s44

Model: cnn_bilstm (786,625 parameters), best dev epoch 9.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 2.43 | 1.89–2.93 | 0.0781 | 0.9963 | 97.56 | 99.72 | 97.56 |
| asvspoof2019_la:eval | 71,237 | 10.83 | 10.59–11.10 | 0.2398 | 0.9515 | 81.19 | 99.69 | 79.28 |
| in_the_wild:test | 31,779 | 24.04 | 23.58–24.50 |  | 0.8201 | 74.13 | 66.46 | 61.43 |
| deep_voice:test | 14,896 | 46.03 | 37.38–56.90 |  | 0.5704 | 38.35 | 90.04 | 33.21 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.6250 | 28.12 | 91.67 | 19.64 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.76 | 1.79 | 0.79 | 2.56 | 2.12 | 2.73 | 5.41 | 2.85 | 1.92 | 2.93 | 27.84 | 35.31 | 10.58 |
