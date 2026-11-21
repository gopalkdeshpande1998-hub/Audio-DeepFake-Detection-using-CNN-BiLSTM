# Results: e2a_lfcc_s42__trimmed

Model: cnn_bilstm (786,625 parameters), best dev epoch 28.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 6.48 | 5.68–7.33 | 0.2165 | 0.9826 | 97.70 | 97.54 | 99.96 |
| asvspoof2019_la:eval | 71,237 | 22.00 | 21.64–22.41 | 0.5434 | 0.8672 | 76.30 | 97.23 | 75.73 |
| in_the_wild:test | 31,779 | 41.58 | 41.04–42.06 |  | 0.5846 | 37.51 | 37.07 | 97.59 |
| deep_voice:test | 14,896 | 49.25 | 34.34–59.58 |  | 0.5202 | 87.22 | 87.61 | 99.47 |
| deep_voice:test (per recording) | 64 | 50.89 |  |  | 0.4598 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.09 | 2.68 | 1.28 | 29.55 | 15.06 | 39.78 | 26.51 | 6.27 | 15.89 | 5.03 | 34.03 | 49.72 | 6.37 |
