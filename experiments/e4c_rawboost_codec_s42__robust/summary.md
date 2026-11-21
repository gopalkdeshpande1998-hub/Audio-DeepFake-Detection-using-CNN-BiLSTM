# Results: e4c_rawboost_codec_s42__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 11.95 | 11.23–12.64 | 0.2868 | 0.9536 | 71.47 | 99.82 | 65.89 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 19.05 | 18.04–19.84 | 0.4727 | 0.8983 | 73.68 | 97.69 | 70.08 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 22.90 | 21.86–23.87 | 0.6232 | 0.8636 | 72.27 | 95.57 | 69.98 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 11.94 | 11.32–12.63 | 0.2892 | 0.9533 | 70.07 | 99.83 | 64.20 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 20.50 | 19.51–21.46 | 0.5666 | 0.8793 | 73.13 | 96.34 | 70.44 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 19.00 | 18.11–19.74 | 0.4993 | 0.8965 | 63.17 | 98.86 | 56.46 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 18.65 | 17.77–19.59 | 0.491 | 0.8963 | 35.23 | 99.96 | 22.30 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 31.74 | 30.92–32.91 | 0.7932 | 0.7514 | 25.69 | 99.54 | 10.89 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 30.80 | 29.96–31.76 | 0.7375 | 0.7597 | 44.19 | 97.72 | 33.83 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 38.61 | 37.58–39.65 | 0.7981 | 0.6845 | 55.63 | 91.03 | 51.88 |
