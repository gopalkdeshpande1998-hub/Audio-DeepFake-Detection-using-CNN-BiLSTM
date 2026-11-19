# e1_cnn_bilstm_mfcc: mean ± std over 3 runs

Runs: `e1_cnn_bilstm_mfcc_s42`, `e1_cnn_bilstm_mfcc_s43`, `e1_cnn_bilstm_mfcc_s44`

| Test set | EER % | min t-DCF | AUC |
|---|---|---|---|
| asvspoof2019_la:dev | 1.36 ± 0.47 | 0.0450 ± 0.0160 | 0.9990 ± 0.0010 |
| asvspoof2019_la:eval | 7.97 ± 0.04 | 0.2100 ± 0.0140 | 0.9740 ± 0.0010 |
| in_the_wild:test | 47.32 ± 4.61 |  | 0.5610 ± 0.0540 |
| deep_voice:test | 48.84 ± 1.32 |  | 0.5250 ± 0.0240 |
| deep_voice:test (per recording) | 52.38 ± 6.58 |  |  |

## asvspoof2019_la:eval: EER per attack (%, mean ± std)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.6 ± 0.2 | 6.1 ± 1.5 | 1.3 ± 0.2 | 3.2 ± 0.6 | 2.3 ± 0.5 | 3.1 ± 0.6 | 2.9 ± 0.3 | 4.2 ± 1.1 | 2.9 ± 0.6 | 1.9 ± 0.6 | 25.8 ± 2.5 | 16.1 ± 2.1 | 6.0 ± 1.5 |
