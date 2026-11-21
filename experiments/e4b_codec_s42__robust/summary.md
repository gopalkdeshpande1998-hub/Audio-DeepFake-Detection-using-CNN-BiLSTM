# Results: e4b_codec_s42__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 4.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 16.70 | 15.71–17.70 | 0.3913 | 0.9190 | 66.99 | 99.82 | 60.50 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 28.05 | 27.05–29.12 | 0.637 | 0.8120 | 49.15 | 99.34 | 39.25 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 31.89 | 30.29–33.02 | 0.8543 | 0.7524 | 53.93 | 93.51 | 48.06 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 17.45 | 16.67–18.19 | 0.4026 | 0.9139 | 65.64 | 99.85 | 58.86 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 28.50 | 27.43–29.66 | 0.7152 | 0.8010 | 52.51 | 96.68 | 44.56 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 27.80 | 26.81–28.98 | 0.6849 | 0.8043 | 72.59 | 92.70 | 72.85 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 41.20 | 40.03–42.31 | 0.8443 | 0.6409 | 18.49 | 99.55 | 2.22 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 38.65 | 37.76–39.62 | 0.859 | 0.6632 | 18.17 | 95.05 | 1.92 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 52.00 | 50.38–53.51 | 1.0 | 0.4672 | 17.97 | 62.46 | 3.96 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 19.90 | 19.18–20.71 | 0.3868 | 0.8802 | 16.76 | 100.00 | 0.13 |
