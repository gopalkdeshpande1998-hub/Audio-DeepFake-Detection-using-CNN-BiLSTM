# Results: e6b_crnn_seres_asp_s42

Model: crnn (1,422,206 parameters), best dev epoch 13.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.09 | 0.91–1.29 | 0.0319 | 0.9993 | 98.92 | 99.88 | 98.91 |
| asvspoof2019_la:eval | 71,237 | 17.73 | 17.42–18.05 | 0.3362 | 0.9149 | 70.09 | 99.90 | 66.71 |
| in_the_wild:test | 31,779 | 71.78 | 71.23–72.51 |  | 0.2463 | 37.30 | 37.02 | 97.89 |
| deep_voice:test | 14,896 | 32.76 | 15.98–52.11 |  | 0.7550 | 89.03 | 88.86 | 100.00 |
| deep_voice:test (per recording) | 64 | 25.00 |  |  | 0.7835 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.38 | 0.08 | 0.07 | 34.05 | 2.05 | 42.47 | 35.96 | 0.42 | 3.19 | 0.95 | 16.40 | 8.48 | 0.38 |
