# Results: e4b_codec_s43__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 46.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.40 |  | 0.0108 | 0.9997 | 99.64 | 99.87 | 99.61 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 8.05 |  | 0.1655 | 0.9788 | 89.44 | 99.50 | 85.30 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 12.26 |  | 0.2899 | 0.9497 | 78.87 | 99.30 | 70.22 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.40 |  | 0.0115 | 0.9997 | 99.46 | 99.89 | 99.33 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 12.70 |  | 0.3009 | 0.9477 | 83.49 | 98.06 | 77.88 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 9.49 |  | 0.1864 | 0.9715 | 88.32 | 99.46 | 83.72 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 55.65 |  | 0.9868 | 0.4128 | 31.99 | 86.42 | 3.03 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 63.40 |  | 0.9996 | 0.3350 | 30.71 | 52.89 | 6.54 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 59.00 |  | 1.0 | 0.3820 | 30.63 | 56.48 | 2.64 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 47.95 |  | 0.8854 | 0.5426 | 31.78 | 100.00 | 2.25 |
