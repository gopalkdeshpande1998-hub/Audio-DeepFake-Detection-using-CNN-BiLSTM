# Results: e1_cnn_bilstm_mfcc_s42__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 27.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.86 |  | 0.0268 | 0.9995 | 98.79 | 99.93 | 98.72 |
| asvspoof2019_la:eval | 71,237 | 7.94 |  | 0.1889 | 0.9740 | 64.34 | 99.94 | 60.27 |
| in_the_wild:test | 31,779 | 41.35 |  |  | 0.6235 | 52.72 | 42.85 | 81.34 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.17 | 3.94 | 1.12 | 2.40 | 1.75 | 2.36 | 2.42 | 3.03 | 2.05 | 1.24 | 27.00 | 17.58 | 4.11 |
