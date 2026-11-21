# Results: e2a_lfcc_s44__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 19.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.06 |  | 0.0019 | 1.0000 | 99.94 | 99.98 | 99.94 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 15.09 |  | 0.3787 | 0.9238 | 70.38 | 70.22 | 99.94 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 41.50 |  | 0.6614 | 0.6681 | 66.63 | 71.34 | 87.23 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.15 |  | 0.0029 | 1.0000 | 99.86 | 99.98 | 99.83 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 15.39 |  | 0.2969 | 0.9173 | 73.55 | 99.59 | 62.36 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 28.05 |  | 0.6269 | 0.8009 | 69.68 | 69.80 | 99.68 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 36.94 |  | 0.7597 | 0.6822 | 62.57 | 81.33 | 60.17 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 43.20 |  | 0.8941 | 0.6010 | 55.69 | 77.74 | 51.17 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 49.86 |  | 0.9905 | 0.5151 | 34.50 | 81.70 | 7.92 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 28.35 |  | 0.6433 | 0.8057 | 30.97 | 100.00 | 1.08 |
