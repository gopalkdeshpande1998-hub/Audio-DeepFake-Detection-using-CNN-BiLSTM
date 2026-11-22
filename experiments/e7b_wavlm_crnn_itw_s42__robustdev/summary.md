# Results: e7b_wavlm_crnn_itw_s42__robustdev

Model: crnn (91,998,587 parameters), best dev epoch 2.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.40 |  | 0.0089 | 0.9999 | 99.61 | 99.83 | 99.61 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 0.45 |  | 0.011 | 0.9998 | 99.58 | 99.74 | 99.65 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 0.85 |  | 0.0213 | 0.9996 | 99.21 | 99.12 | 99.76 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.40 |  | 0.0099 | 0.9999 | 99.62 | 99.83 | 99.63 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 1.71 |  | 0.0527 | 0.9980 | 96.18 | 94.96 | 99.83 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 1.45 |  | 0.043 | 0.9988 | 98.82 | 98.75 | 99.57 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 4.20 |  | 0.0845 | 0.9845 | 84.38 | 100.00 | 77.62 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 7.20 |  | 0.155 | 0.9813 | 41.54 | 100.00 | 16.23 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 7.90 |  | 0.1979 | 0.9745 | 33.01 | 100.00 | 4.00 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 7.64 |  | 0.1781 | 0.9795 | 43.37 | 100.00 | 18.85 |
