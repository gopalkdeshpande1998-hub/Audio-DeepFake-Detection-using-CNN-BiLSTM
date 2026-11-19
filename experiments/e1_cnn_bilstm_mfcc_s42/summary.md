# Results: e1_cnn_bilstm_mfcc_s42

Model: cnn_bilstm (786,625 parameters), best dev epoch 27.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.86 | 0.028 | 0.9995 | 99.13 | 99.90 | 99.13 |
| asvspoof2019_la:eval | 71,237 | 8.01 | 0.1939 | 0.9744 | 68.45 | 99.93 | 64.87 |
| in_the_wild:test | 31,779 | 42.46 |  | 0.6086 | 52.18 | 42.42 | 80.02 |
| deep_voice:test | 14,896 | 50.16 |  | 0.5020 | 65.43 | 88.29 | 69.75 |
| deep_voice:test (per recording) | 64 | 59.82 |  | 0.4710 | 79.69 | 87.72 | 89.29 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.39 | 4.35 | 1.14 | 2.58 | 1.71 | 2.57 | 2.50 | 3.24 | 2.19 | 1.26 | 26.09 | 17.58 | 4.31 |
