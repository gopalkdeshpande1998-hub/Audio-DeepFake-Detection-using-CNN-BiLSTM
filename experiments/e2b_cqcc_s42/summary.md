# Results: e2b_cqcc_s42

Model: cnn_bilstm (786,625 parameters), best dev epoch 2.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 2.40 | 2.13–2.67 | 0.0716 | 0.9972 | 97.62 | 99.72 | 97.62 |
| asvspoof2019_la:eval | 71,237 | 10.79 | 10.56–11.09 | 0.2363 | 0.9514 | 83.97 | 99.62 | 82.44 |
| in_the_wild:test | 31,779 | 29.69 | 29.21–30.17 |  | 0.7732 | 66.33 | 53.07 | 81.66 |
| deep_voice:test | 14,896 | 43.70 | 30.00–61.81 |  | 0.5887 | 58.32 | 90.05 | 58.88 |
| deep_voice:test (per recording) | 64 | 27.68 |  |  | 0.6763 | 64.06 | 94.59 | 62.50 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.71 | 2.44 | 0.22 | 1.55 | 0.57 | 2.99 | 2.05 | 3.46 | 0.75 | 3.66 | 27.98 | 39.82 | 9.77 |
