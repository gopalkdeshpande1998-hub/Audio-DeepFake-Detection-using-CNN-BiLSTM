# Results: e4d_full_aug_s42__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 18.00 | 17.26–18.70 | 0.4355 | 0.9078 | 78.06 | 97.55 | 75.57 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 19.16 | 18.26–19.90 | 0.4666 | 0.8944 | 79.61 | 96.01 | 78.81 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 23.50 | 22.61–24.39 | 0.5981 | 0.8518 | 69.18 | 96.31 | 65.52 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 18.20 | 17.38–18.89 | 0.4357 | 0.9072 | 78.02 | 97.60 | 75.48 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 23.10 | 21.96–23.97 | 0.5688 | 0.8546 | 78.78 | 93.42 | 80.18 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 21.95 | 20.93–22.68 | 0.5123 | 0.8739 | 70.21 | 97.67 | 65.82 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 26.71 | 25.83–27.45 | 0.56 | 0.8017 | 65.91 | 97.03 | 60.97 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 25.65 | 24.78–26.53 | 0.5501 | 0.8210 | 57.84 | 98.93 | 49.96 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 25.85 | 24.89–26.70 | 0.5838 | 0.8221 | 78.58 | 90.67 | 82.83 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 20.30 | 19.50–21.06 | 0.4749 | 0.8634 | 78.68 | 95.78 | 77.84 |
