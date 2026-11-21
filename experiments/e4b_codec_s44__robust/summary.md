# Results: e4b_codec_s44__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 13.80 | 13.19–14.63 | 0.3373 | 0.9387 | 68.49 | 99.82 | 62.31 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 20.55 | 19.62–21.46 | 0.5518 | 0.8916 | 64.90 | 97.88 | 59.17 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 26.65 | 25.66–27.70 | 0.7017 | 0.8184 | 45.99 | 98.54 | 35.72 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 14.30 | 13.55–15.18 | 0.3458 | 0.9359 | 67.52 | 99.84 | 61.13 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 24.96 | 24.04–26.00 | 0.6671 | 0.8406 | 61.08 | 96.69 | 55.19 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 22.40 | 21.18–23.52 | 0.5938 | 0.8708 | 62.28 | 97.50 | 56.18 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 39.30 | 38.30–40.32 | 0.8041 | 0.6574 | 23.60 | 100.00 | 8.33 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 58.96 | 58.02–60.02 | 0.9785 | 0.3816 | 17.08 | 100.00 | 0.51 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 69.65 | 68.51–70.62 | 0.9998 | 0.2412 | 16.65 | 0.00 | 0.00 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 57.45 | 56.34–58.42 | 0.9619 | 0.4193 | 17.31 | 100.00 | 0.79 |
