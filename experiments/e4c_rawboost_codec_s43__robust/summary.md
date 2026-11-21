# Results: e4c_rawboost_codec_s43__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 13.14 | 12.53–13.79 | 0.2882 | 0.9470 | 74.65 | 99.73 | 69.78 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 19.49 | 18.56–20.30 | 0.4887 | 0.8932 | 67.40 | 98.57 | 61.79 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 24.59 | 23.55–25.46 | 0.6603 | 0.8403 | 76.58 | 93.55 | 77.22 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 13.29 | 12.61–14.00 | 0.2969 | 0.9463 | 73.59 | 99.80 | 68.45 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 23.55 | 22.44–24.42 | 0.6083 | 0.8527 | 81.42 | 92.29 | 84.79 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 19.16 | 18.36–20.10 | 0.5261 | 0.8898 | 64.78 | 98.38 | 58.71 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 26.50 | 25.50–27.39 | 0.6611 | 0.8215 | 41.85 | 99.48 | 30.39 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 35.15 | 34.05–36.24 | 0.811 | 0.7166 | 53.20 | 93.86 | 46.91 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 23.55 | 22.62–24.54 | 0.6609 | 0.8333 | 74.52 | 94.73 | 73.52 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 27.15 | 26.10–28.17 | 0.7159 | 0.8019 | 57.64 | 96.10 | 51.26 |
