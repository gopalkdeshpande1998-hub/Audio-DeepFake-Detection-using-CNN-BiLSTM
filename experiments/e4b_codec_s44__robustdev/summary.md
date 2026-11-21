# Results: e4b_codec_s44__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.41 |  | 0.0137 | 0.9998 | 99.58 | 99.83 | 99.57 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 10.71 |  | 0.2761 | 0.9560 | 89.15 | 95.37 | 88.77 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 17.09 |  | 0.3857 | 0.9083 | 77.34 | 98.06 | 68.90 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.55 |  | 0.0152 | 0.9998 | 99.32 | 99.78 | 99.24 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 16.94 |  | 0.4191 | 0.9152 | 82.84 | 92.36 | 82.21 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 13.10 |  | 0.318 | 0.9433 | 86.74 | 94.55 | 85.95 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 34.65 |  | 0.7503 | 0.7170 | 37.64 | 100.00 | 10.65 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 53.16 |  | 0.96 | 0.4637 | 30.89 | 100.00 | 0.97 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 59.40 |  | 0.9989 | 0.3645 | 30.21 | 0.00 | 0.00 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 50.65 |  | 0.8754 | 0.5157 | 32.60 | 100.00 | 3.42 |
