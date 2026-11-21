# Results: e1_cnn_bilstm_mfcc_s43__trimmed

Model: cnn_bilstm (786,625 parameters), best dev epoch 13.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 17.78 | 16.93–18.82 | 0.515 | 0.9099 | 91.56 | 95.54 | 95.03 |
| asvspoof2019_la:eval | 71,237 | 27.44 | 26.91–27.99 | 0.7932 | 0.8084 | 77.64 | 95.27 | 78.98 |
| in_the_wild:test | 31,779 | 55.33 | 54.83–55.93 |  | 0.4266 | 44.95 | 33.94 | 50.77 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 18.01 | 16.18 | 14.98 | 29.83 | 22.10 | 35.74 | 29.69 | 36.41 | 27.48 | 20.60 | 39.11 | 37.18 | 21.33 |
