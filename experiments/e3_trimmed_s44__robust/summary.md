# Results: e3_trimmed_s44__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 29.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 21.50 | 20.69–22.27 | 0.3607 | 0.8673 | 68.36 | 99.98 | 62.05 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 33.00 | 31.87–34.09 | 0.7853 | 0.7479 | 83.31 | 83.37 | 99.90 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 31.60 | 30.52–32.81 | 0.8288 | 0.7447 | 83.33 | 83.34 | 99.98 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 22.19 | 21.35–22.94 | 0.361 | 0.8626 | 68.13 | 99.97 | 61.79 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 33.51 | 32.30–34.46 | 0.8817 | 0.7273 | 83.37 | 83.38 | 99.97 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 38.94 | 37.80–40.19 | 0.8483 | 0.6924 | 83.35 | 83.35 | 100.00 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 24.20 | 23.31–25.05 | 0.5448 | 0.8280 | 25.14 | 100.00 | 10.18 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 38.14 | 37.02–38.98 | 0.8518 | 0.6649 | 17.07 | 100.00 | 0.50 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 35.21 | 34.36–36.25 | 0.8115 | 0.6952 | 17.73 | 100.00 | 1.29 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 23.75 | 22.71–24.82 | 0.7041 | 0.8316 | 16.79 | 100.00 | 0.17 |
