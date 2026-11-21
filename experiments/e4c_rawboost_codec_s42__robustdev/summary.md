# Results: e4c_rawboost_codec_s42__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 1.04 |  | 0.0226 | 0.9990 | 98.96 | 99.59 | 98.92 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 10.30 |  | 0.2741 | 0.9627 | 89.85 | 94.02 | 91.26 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 14.71 |  | 0.3699 | 0.9337 | 85.44 | 91.79 | 86.90 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 1.09 |  | 0.0249 | 0.9991 | 98.87 | 99.67 | 98.70 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 14.30 |  | 0.3796 | 0.9350 | 85.86 | 92.80 | 86.45 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 11.95 |  | 0.2968 | 0.9547 | 84.17 | 97.88 | 79.03 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 21.51 |  | 0.52 | 0.8754 | 47.58 | 99.83 | 24.94 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 33.61 |  | 0.7976 | 0.7182 | 40.24 | 98.54 | 14.59 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 34.30 |  | 0.8244 | 0.7018 | 51.06 | 92.07 | 32.68 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 33.80 |  | 0.6994 | 0.7231 | 66.18 | 80.74 | 67.68 |
