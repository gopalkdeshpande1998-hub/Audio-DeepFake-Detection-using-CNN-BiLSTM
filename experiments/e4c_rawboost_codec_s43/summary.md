# Results: e4c_rawboost_codec_s43

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.96 | 1.60–2.29 | 0.0572 | 0.9980 | 98.03 | 99.77 | 98.03 |
| asvspoof2019_la:eval | 71,237 | 13.41 | 13.10–13.71 | 0.312 | 0.9449 | 73.04 | 99.77 | 70.10 |
| in_the_wild:test | 31,779 | 44.70 | 44.12–45.27 |  | 0.5828 | 47.08 | 39.76 | 82.13 |
| deep_voice:test | 14,896 | 33.35 | 18.40–54.69 |  | 0.7001 | 81.79 | 90.66 | 88.28 |
| deep_voice:test (per recording) | 64 | 25.00 |  |  | 0.8013 | 93.75 | 94.83 | 98.21 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.10 | 0.07 | 0.20 | 30.89 | 9.20 | 31.04 | 11.74 | 2.34 | 9.04 | 3.25 | 13.50 | 14.59 | 1.32 |
