# Results: e1_cnn_bilstm_mfcc_s44

Model: cnn_bilstm (786,625 parameters), best dev epoch 17.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.42 | 0.0476 | 0.9988 | 98.58 | 99.84 | 98.57 |
| asvspoof2019_la:eval | 71,237 | 7.97 | 0.2202 | 0.9727 | 66.20 | 99.85 | 62.41 |
| in_the_wild:test | 31,779 | 47.89 |  | 0.5726 | 50.69 | 39.93 | 64.71 |
| deep_voice:test | 14,896 | 47.53 |  | 0.5500 | 62.72 | 88.14 | 66.32 |
| deep_voice:test (per recording) | 64 | 50.00 |  | 0.5312 | 70.31 | 89.36 | 75.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.77 | 6.78 | 1.44 | 3.32 | 2.53 | 3.03 | 2.93 | 4.07 | 3.17 | 1.85 | 23.12 | 17.09 | 6.65 |
