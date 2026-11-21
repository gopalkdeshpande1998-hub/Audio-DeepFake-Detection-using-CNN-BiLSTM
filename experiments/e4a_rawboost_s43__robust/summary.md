# Results: e4a_rawboost_s43__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 40.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 13.74 | 13.05–14.54 | 0.3177 | 0.9419 | 63.73 | 99.98 | 56.49 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 25.80 | 24.95–27.06 | 0.596 | 0.8364 | 41.12 | 99.93 | 29.37 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 31.20 | 30.27–32.28 | 0.748 | 0.7610 | 75.90 | 88.84 | 81.29 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 13.80 | 13.17–14.59 | 0.3182 | 0.9414 | 62.66 | 100.00 | 55.19 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 29.15 | 28.19–30.12 | 0.6544 | 0.7924 | 55.25 | 97.79 | 47.37 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 29.90 | 28.75–30.93 | 0.7047 | 0.7779 | 60.78 | 94.84 | 55.99 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 18.41 | 17.60–19.40 | 0.49 | 0.9089 | 39.61 | 100.00 | 27.54 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 30.64 | 29.74–31.84 | 0.7442 | 0.7712 | 24.60 | 99.79 | 9.55 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 35.05 | 34.07–36.03 | 0.7617 | 0.7014 | 24.95 | 99.80 | 9.98 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 23.15 | 22.23–24.09 | 0.5941 | 0.8442 | 42.52 | 99.43 | 31.22 |
