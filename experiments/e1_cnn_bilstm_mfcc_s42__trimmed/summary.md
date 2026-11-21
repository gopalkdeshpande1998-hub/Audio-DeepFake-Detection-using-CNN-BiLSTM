# Results: e1_cnn_bilstm_mfcc_s42__trimmed

Model: cnn_bilstm (786,625 parameters), best dev epoch 27.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 15.34 | 14.38–16.16 | 0.4597 | 0.9283 | 93.02 | 95.96 | 96.27 |
| asvspoof2019_la:eval | 71,237 | 28.41 | 27.81–29.00 | 0.7863 | 0.8033 | 70.92 | 95.69 | 70.76 |
| in_the_wild:test | 31,779 | 45.81 | 45.29–46.33 |  | 0.5537 | 49.21 | 40.85 | 81.70 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15.36 | 17.79 | 15.10 | 31.56 | 21.06 | 38.67 | 32.81 | 38.56 | 29.73 | 15.46 | 37.26 | 40.22 | 16.91 |
