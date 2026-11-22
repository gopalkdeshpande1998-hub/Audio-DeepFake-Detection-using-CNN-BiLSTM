# P3.4 Score-level fusion

Logistic regression on ASVspoof **dev** scores, one fusion per seed (seeds [42]); EER in %, mean ± std when there are several seeds. Test sets are never used to fit the fusion.

Spectrogram member chosen by mean degraded-dev EER: **e4a_rawboost** (18.43%; candidates: e4a_rawboost 18.43, e5a_lcnn 19.52, e6b_crnn_seres_asp 30.49).

| System | ASVspoof eval | eval min t-DCF | In-the-Wild | DEEP-VOICE (recording) | degraded eval, mean of 10 |
|---|---|---|---|---|---|
| e4a_rawboost | 15.62 | 0.370 | 35.98 | 50.00 | 26.42 |
| e7a_wavlm_crnn | 6.22 | 0.125 | 10.67 | 12.50 | – |
| **e4a_rawboost + e7a_wavlm_crnn** | 7.34 | 0.140 | 17.08 | 35.71 | – |

Full fusion vs best single member (positive = fusion better, EER points): ASVspoof eval -1.13; In-the-Wild -6.41; DEEP-VOICE (recording) -23.21.
