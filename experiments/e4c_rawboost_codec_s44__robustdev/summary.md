# Results: e4c_rawboost_codec_s44__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 19.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 2.20 |  | 0.0608 | 0.9978 | 97.87 | 98.93 | 98.01 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 17.64 |  | 0.3651 | 0.9149 | 79.40 | 96.12 | 73.44 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 21.10 |  | 0.4782 | 0.8781 | 79.24 | 87.34 | 82.16 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 2.34 |  | 0.0631 | 0.9976 | 97.60 | 99.08 | 97.47 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 22.55 |  | 0.5253 | 0.8622 | 78.08 | 79.95 | 91.56 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 19.56 |  | 0.4183 | 0.8989 | 80.65 | 90.32 | 80.95 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 22.10 |  | 0.5346 | 0.8552 | 55.20 | 98.76 | 36.26 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 35.64 |  | 0.8599 | 0.6882 | 57.22 | 86.85 | 45.61 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 49.59 |  | 0.9586 | 0.5080 | 48.41 | 74.91 | 39.22 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 29.20 |  | 0.7392 | 0.7908 | 75.18 | 78.62 | 88.51 |
