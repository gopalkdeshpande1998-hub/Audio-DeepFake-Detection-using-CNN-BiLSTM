# Results: e4c_rawboost_codec_s44__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 19.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 12.35 | 11.70–13.07 | 0.293 | 0.9479 | 78.47 | 99.45 | 74.58 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 20.30 | 19.27–21.21 | 0.5105 | 0.8875 | 69.28 | 97.91 | 64.52 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 24.15 | 23.18–25.00 | 0.6493 | 0.8435 | 76.49 | 93.83 | 76.85 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 12.35 | 11.74–13.02 | 0.2964 | 0.9473 | 77.19 | 99.47 | 73.02 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 24.84 | 23.95–25.77 | 0.6589 | 0.8359 | 83.78 | 89.86 | 90.78 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 21.69 | 20.75–22.74 | 0.5929 | 0.8686 | 75.90 | 95.38 | 74.70 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 27.70 | 26.61–28.82 | 0.692 | 0.8197 | 36.06 | 99.45 | 23.42 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 43.30 | 42.21–44.45 | 0.9115 | 0.6274 | 41.42 | 93.15 | 32.07 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 62.99 | 61.92–63.83 | 0.9949 | 0.3571 | 31.14 | 79.19 | 23.58 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 29.55 | 28.31–30.47 | 0.818 | 0.7707 | 78.91 | 89.69 | 84.40 |
