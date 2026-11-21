# Results: e4c_rawboost_codec_s43__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 1.90 |  | 0.0546 | 0.9982 | 98.23 | 99.11 | 98.35 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 13.29 |  | 0.324 | 0.9499 | 83.43 | 97.42 | 78.33 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 16.65 |  | 0.4086 | 0.9205 | 84.85 | 88.29 | 90.26 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 2.04 |  | 0.0623 | 0.9979 | 97.70 | 99.21 | 97.49 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 17.46 |  | 0.4414 | 0.9069 | 82.36 | 83.90 | 92.47 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 14.80 |  | 0.3723 | 0.9379 | 80.32 | 96.84 | 74.22 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 28.10 |  | 0.7022 | 0.8034 | 51.93 | 97.55 | 31.93 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 36.94 |  | 0.8739 | 0.6756 | 58.41 | 83.59 | 50.28 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 30.39 |  | 0.7694 | 0.7470 | 66.33 | 86.91 | 60.93 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 35.80 |  | 0.8115 | 0.6961 | 55.20 | 89.80 | 40.39 |
