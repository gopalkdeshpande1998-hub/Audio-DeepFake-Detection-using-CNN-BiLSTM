# Results: e3_trimmed_s42__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 10.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.16 |  | 0.0048 | 0.9999 | 99.83 | 99.93 | 99.83 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 27.00 |  | 0.7084 | 0.8024 | 69.83 | 69.84 | 99.94 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 39.05 |  | 0.7006 | 0.6742 | 69.79 | 69.79 | 100.00 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.35 |  | 0.007 | 0.9998 | 99.64 | 99.93 | 99.55 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 34.55 |  | 0.7124 | 0.7075 | 67.11 | 69.23 | 95.19 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 40.20 |  | 0.9992 | 0.6189 | 69.79 | 69.79 | 100.00 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 22.59 |  | 0.5951 | 0.8301 | 32.81 | 100.00 | 3.72 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 50.26 |  | 0.9813 | 0.4914 | 30.23 | 100.00 | 0.02 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 55.00 |  | 0.9981 | 0.3944 | 30.21 | 0.00 | 0.00 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 28.14 |  | 0.6015 | 0.7971 | 30.39 | 100.00 | 0.26 |
