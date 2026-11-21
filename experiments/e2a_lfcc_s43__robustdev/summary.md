# Results: e2a_lfcc_s43__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 29.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.00 |  | 0.0 | 1.0000 | 99.98 | 100.00 | 99.98 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 29.25 |  | 0.7456 | 0.8029 | 69.88 | 69.85 | 100.00 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 36.35 |  | 0.7485 | 0.7001 | 69.74 | 69.90 | 99.50 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.05 |  | 0.0009 | 1.0000 | 99.86 | 100.00 | 99.81 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 13.45 |  | 0.2438 | 0.9218 | 84.11 | 88.30 | 89.03 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 19.06 |  | 0.5088 | 0.8775 | 70.29 | 70.17 | 99.89 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 38.84 |  | 0.965 | 0.6266 | 58.66 | 83.12 | 51.15 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 55.50 |  | 1.0 | 0.3876 | 48.82 | 65.65 | 55.93 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 49.09 |  | 0.9988 | 0.5019 | 38.04 | 65.91 | 23.23 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 31.90 |  | 0.66 | 0.7446 | 32.15 | 100.00 | 2.77 |
