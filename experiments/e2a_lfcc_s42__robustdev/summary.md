# Results: e2a_lfcc_s42__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 28.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.09 |  | 0.0011 | 1.0000 | 99.92 | 99.98 | 99.91 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 14.80 |  | 0.3693 | 0.9301 | 70.63 | 70.40 | 99.94 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 20.45 |  | 0.515 | 0.8432 | 69.74 | 69.83 | 99.74 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.15 |  | 0.0028 | 1.0000 | 99.80 | 99.98 | 99.74 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 13.80 |  | 0.2415 | 0.9102 | 77.66 | 79.29 | 92.01 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 27.95 |  | 0.6244 | 0.7718 | 69.79 | 69.79 | 100.00 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 43.99 |  | 0.9989 | 0.5582 | 56.00 | 74.63 | 55.97 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 54.34 |  | 0.9998 | 0.4056 | 57.21 | 66.74 | 77.10 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 48.55 |  | 0.9996 | 0.4819 | 65.44 | 68.90 | 92.01 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 51.60 |  | 0.8075 | 0.4991 | 34.03 | 100.00 | 5.48 |
