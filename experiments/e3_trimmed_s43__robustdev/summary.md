# Results: e3_trimmed_s43__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 33.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.15 |  | 0.0038 | 1.0000 | 99.86 | 99.94 | 99.87 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 46.15 |  | 0.9985 | 0.5481 | 69.79 | 69.79 | 100.00 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 34.64 |  | 0.8376 | 0.6979 | 69.79 | 69.79 | 100.00 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.30 |  | 0.0069 | 0.9998 | 99.62 | 99.93 | 99.52 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 31.01 |  | 0.6787 | 0.7347 | 69.52 | 69.72 | 99.57 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 36.46 |  | 0.7394 | 0.6965 | 69.79 | 69.79 | 100.00 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 42.95 |  | 0.9407 | 0.6123 | 38.08 | 88.71 | 12.92 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 46.00 |  | 0.9996 | 0.5379 | 32.76 | 65.62 | 7.68 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 58.65 |  | 1.0 | 0.4021 | 33.19 | 57.04 | 17.27 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 53.50 |  | 0.9397 | 0.4733 | 33.97 | 96.98 | 5.56 |
