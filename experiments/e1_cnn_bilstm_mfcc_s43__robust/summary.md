# Results: e1_cnn_bilstm_mfcc_s43__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 13.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 8.30 | 7.72–8.90 | 0.2239 | 0.9718 | 74.65 | 99.70 | 69.79 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 12.20 | 11.49–12.91 | 0.3567 | 0.9455 | 85.92 | 97.83 | 85.00 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 13.50 | 12.83–14.14 | 0.3064 | 0.9466 | 44.39 | 99.76 | 33.36 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 11.84 | 10.98–12.69 | 0.3552 | 0.9459 | 48.46 | 99.46 | 38.37 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 24.55 | 23.08–26.19 | 0.8257 | 0.8260 | 44.50 | 93.79 | 35.78 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 76.56 | 75.54–77.72 | 0.9997 | 0.1921 | 37.15 | 70.52 | 42.26 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 49.30 | 48.36–50.49 | 0.9839 | 0.5115 | 49.63 | 84.11 | 48.79 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 55.55 | 54.48–56.46 | 0.9977 | 0.4057 | 32.09 | 84.26 | 22.78 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 68.90 | 67.96–70.01 | 0.9986 | 0.2461 | 17.15 | 89.47 | 0.68 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 60.56 | 59.63–61.53 | 0.9991 | 0.3552 | 57.99 | 78.77 | 67.90 |
