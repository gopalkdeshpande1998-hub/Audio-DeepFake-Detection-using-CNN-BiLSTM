# Results: e4d_full_aug_s43__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 17.60 | 16.69–18.63 | 0.4589 | 0.9138 | 76.07 | 97.75 | 72.97 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 18.25 | 17.38–19.24 | 0.4704 | 0.9068 | 79.41 | 96.42 | 78.20 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 21.95 | 20.95–22.93 | 0.6159 | 0.8649 | 76.28 | 95.09 | 75.43 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 17.70 | 16.79–18.62 | 0.4598 | 0.9131 | 76.18 | 97.79 | 73.07 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 22.75 | 21.71–23.88 | 0.5823 | 0.8612 | 83.76 | 91.32 | 88.98 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 20.70 | 19.79–21.48 | 0.5194 | 0.8862 | 77.74 | 95.77 | 76.68 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 30.55 | 29.32–31.77 | 0.8098 | 0.7761 | 50.73 | 95.15 | 43.09 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 37.15 | 35.87–38.34 | 0.9198 | 0.6882 | 41.53 | 91.55 | 32.89 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 45.40 | 44.42–46.26 | 0.8526 | 0.6123 | 58.13 | 83.71 | 61.79 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 34.60 | 33.53–35.70 | 0.9148 | 0.7254 | 53.62 | 92.00 | 48.58 |
