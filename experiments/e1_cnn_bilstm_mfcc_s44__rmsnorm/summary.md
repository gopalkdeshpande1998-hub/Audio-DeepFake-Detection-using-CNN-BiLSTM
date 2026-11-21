# Results: e1_cnn_bilstm_mfcc_s44__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 17.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.53 |  | 0.0488 | 0.9986 | 97.90 | 99.85 | 97.81 |
| asvspoof2019_la:eval | 71,237 | 7.82 |  | 0.2174 | 0.9729 | 59.99 | 99.86 | 55.46 |
| in_the_wild:test | 31,779 | 47.35 |  |  | 0.5815 | 51.33 | 40.60 | 66.77 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.83 | 5.76 | 1.36 | 3.46 | 2.57 | 3.17 | 3.09 | 4.13 | 3.09 | 1.92 | 22.99 | 17.05 | 6.47 |
