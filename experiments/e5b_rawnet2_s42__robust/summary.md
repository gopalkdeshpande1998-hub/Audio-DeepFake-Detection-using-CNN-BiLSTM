# Results: e5b_rawnet2_s42__robust

Model: rawnet2 (17,620,385 parameters), best dev epoch 12.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 18.70 | 17.95–19.62 | 0.5016 | 0.8972 | 64.55 | 98.58 | 58.31 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 19.05 | 18.17–19.99 | 0.5092 | 0.8947 | 65.20 | 98.47 | 59.16 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 20.20 | 19.41–21.06 | 0.53 | 0.8836 | 67.91 | 97.88 | 62.86 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 18.85 | 17.95–19.60 | 0.4947 | 0.8979 | 64.74 | 98.60 | 58.52 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 19.60 | 18.64–20.43 | 0.5272 | 0.8842 | 76.22 | 96.48 | 74.18 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 24.15 | 23.09–25.18 | 0.6272 | 0.8426 | 61.37 | 97.24 | 55.22 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 28.00 | 26.96–28.91 | 0.7034 | 0.7862 | 42.73 | 98.45 | 31.79 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 39.84 | 38.73–40.93 | 0.8148 | 0.6561 | 33.79 | 98.54 | 20.87 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 32.10 | 31.06–33.09 | 0.7801 | 0.7501 | 28.68 | 99.05 | 14.58 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 38.31 | 37.36–39.44 | 0.8694 | 0.6650 | 56.79 | 90.33 | 53.93 |
