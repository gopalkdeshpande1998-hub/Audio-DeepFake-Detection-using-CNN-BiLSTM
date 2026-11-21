# Results: e4b_codec_s42

Model: cnn_bilstm (786,625 parameters), best dev epoch 4.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.14 | 0.94–1.29 | 0.0287 | 0.9992 | 98.86 | 99.87 | 98.86 |
| asvspoof2019_la:eval | 71,237 | 16.59 | 16.13–16.99 | 0.3904 | 0.9188 | 64.42 | 99.84 | 60.41 |
| in_the_wild:test | 31,779 | 47.83 | 47.24–48.46 |  | 0.5188 | 34.96 | 33.11 | 73.46 |
| deep_voice:test | 14,896 | 28.25 | 15.99–50.11 |  | 0.7452 | 83.00 | 90.97 | 89.46 |
| deep_voice:test (per recording) | 64 | 23.21 |  |  | 0.8594 | 90.62 | 90.32 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 0.02 | 0.10 | 25.86 | 10.62 | 39.16 | 21.39 | 1.14 | 8.75 | 1.44 | 18.66 | 28.61 | 0.51 |
