# Results: e1_cnn_bilstm_mfcc_s44__trimmed

Model: cnn_bilstm (786,625 parameters), best dev epoch 17.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 17.15 | 16.39–17.75 | 0.4907 | 0.9189 | 90.22 | 96.50 | 92.46 |
| asvspoof2019_la:eval | 71,237 | 27.18 | 26.63–27.70 | 0.7907 | 0.8112 | 69.91 | 96.13 | 69.23 |
| in_the_wild:test | 31,779 | 50.56 | 49.96–51.10 |  | 0.5186 | 48.50 | 38.57 | 64.95 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 18.26 | 19.51 | 15.32 | 30.55 | 21.41 | 35.85 | 28.55 | 31.46 | 30.12 | 17.32 | 34.15 | 43.65 | 19.98 |
