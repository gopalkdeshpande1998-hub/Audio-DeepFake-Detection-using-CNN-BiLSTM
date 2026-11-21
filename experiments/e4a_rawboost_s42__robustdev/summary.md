# Results: e4a_rawboost_s42__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.44 |  | 0.0138 | 0.9994 | 99.56 | 99.80 | 99.57 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 23.10 |  | 0.6644 | 0.8326 | 77.76 | 87.82 | 79.11 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 23.90 |  | 0.4835 | 0.8407 | 72.10 | 72.45 | 96.86 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.50 |  | 0.0128 | 0.9994 | 99.44 | 99.85 | 99.35 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 22.70 |  | 0.5002 | 0.8581 | 78.16 | 82.47 | 87.25 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 24.90 |  | 0.5213 | 0.8403 | 72.66 | 72.60 | 97.71 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 15.94 |  | 0.4373 | 0.9131 | 63.84 | 99.03 | 48.66 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 29.25 |  | 0.8045 | 0.7754 | 41.28 | 95.76 | 16.60 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 37.39 |  | 0.8009 | 0.6924 | 33.70 | 94.25 | 5.32 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 27.20 |  | 0.6804 | 0.8126 | 54.97 | 97.67 | 36.34 |
