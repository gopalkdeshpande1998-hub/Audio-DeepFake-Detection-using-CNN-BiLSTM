# Results: e1_cnn_bilstm_mfcc_s43

Model: cnn_bilstm (786,625 parameters), best dev epoch 13.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.81 | 0.0601 | 0.9983 | 98.18 | 99.79 | 98.18 |
| asvspoof2019_la:eval | 71,237 | 7.94 | 0.216 | 0.9735 | 72.93 | 99.84 | 69.93 |
| in_the_wild:test | 31,779 | 51.62 |  | 0.5016 | 48.27 | 37.15 | 56.57 |
| deep_voice:test | 14,896 | 48.83 |  | 0.5222 | 59.02 | 87.46 | 62.07 |
| deep_voice:test (per recording) | 64 | 47.32 |  | 0.5580 | 64.06 | 86.67 | 69.64 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.65 | 7.06 | 1.36 | 3.70 | 2.62 | 3.77 | 3.13 | 5.33 | 3.28 | 2.48 | 28.17 | 13.65 | 7.08 |
