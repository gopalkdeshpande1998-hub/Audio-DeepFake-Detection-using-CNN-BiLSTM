# Results: e6b_crnn_seres_asp_s42__robust

Model: crnn (1,422,206 parameters), best dev epoch 13.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 17.64 | 16.98–18.46 | 0.3388 | 0.9152 | 72.10 | 99.84 | 66.63 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 40.35 | 38.79–42.56 | 1.0 | 0.6356 | 83.32 | 83.38 | 99.90 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 30.60 | 29.66–31.73 | 0.8678 | 0.7397 | 83.08 | 83.41 | 99.49 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 17.90 | 17.22–18.60 | 0.3447 | 0.9159 | 70.77 | 99.86 | 65.01 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 36.05 | 34.93–37.94 | 0.9969 | 0.6725 | 83.12 | 83.57 | 99.27 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 36.10 | 34.97–38.28 | 0.9998 | 0.6734 | 83.36 | 83.37 | 99.98 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 38.15 | 37.03–39.27 | 0.9477 | 0.6430 | 59.53 | 89.87 | 57.98 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 45.85 | 44.66–46.97 | 0.9793 | 0.5673 | 79.33 | 83.66 | 93.47 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 63.35 | 62.23–64.41 | 0.9966 | 0.3263 | 57.58 | 78.63 | 67.42 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 45.76 | 44.39–47.31 | 0.9814 | 0.5637 | 80.37 | 83.68 | 94.99 |
