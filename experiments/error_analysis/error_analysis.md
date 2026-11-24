# Error analysis of the merged detector (P4.6)

Produced by `scripts/evaluation/error_analysis.py` from the stored per-utterance scores; decision: FAKE if merged P(fake) >= 0.5. In-the-Wild is contaminated for VoiceGuard v2 (trained on most of it).

## ASVspoof eval

n = 71,237 · accuracy 81.4% · real → FAKE 11.2% · fake → REAL 19.4% · EER 14.26%

**Accuracy by how many members share the final verdict**

| members agreeing | n | accuracy % |
|---|---|---|
| 1 of 3 | 8,981 | 76.7 |
| 2 of 3 | 37,359 | 73.8 |
| 3 of 3 | 24,897 | 94.6 |

**Accuracy by confidence of the merged probability**

| confidence | n | accuracy % |
|---|---|---|
| very low (50±10%) | 15,520 | 47.8 |
| low (60–70% / 30–40%) | 11,829 | 81.5 |
| medium (70–80% / 20–30%) | 15,490 | 86.5 |
| high (80–90% / 10–20%) | 16,425 | 95.4 |
| very high (≥90% / ≤10%) | 11,973 | 99.1 |

**Per attack** (fake → REAL at 0.5; EER against all bona fide eval)

| attack | n | fake → REAL % | merged EER % | E7a EER % |
|---|---|---|---|---|
| A07 | 4,914 | 16.5 | 13.41 | 3.42 |
| A08 | 4,914 | 0.1 | 0.65 | 0.15 |
| A09 | 4,914 | 0.0 | 0.08 | 0.06 |
| A10 | 4,914 | 70.9 | 33.57 | 31.11 |
| A11 | 4,914 | 26.6 | 17.99 | 6.05 |
| A12 | 4,914 | 17.5 | 13.65 | 7.29 |
| A13 | 4,914 | 2.9 | 5.37 | 0.20 |
| A14 | 4,914 | 5.7 | 8.61 | 0.47 |
| A15 | 4,914 | 4.6 | 7.04 | 1.79 |
| A16 | 4,914 | 15.2 | 12.70 | 1.67 |
| A17 | 4,914 | 8.9 | 9.95 | 0.16 |
| A18 | 4,914 | 51.6 | 21.61 | 1.77 |
| A19 | 4,914 | 32.0 | 14.26 | 0.71 |

**By clip duration**

| duration | n | accuracy % | real → FAKE % | fake → REAL % |
|---|---|---|---|---|
| < 2 s | 16,631 | 83.1 | 14.7 | 16.9 |
| 2–3 s | 23,117 | 82.3 | 11.5 | 18.5 |
| 3–4 s | 15,644 | 79.6 | 11.0 | 22.3 |
| 4–6 s | 12,275 | 80.9 | 11.4 | 20.2 |
| ≥ 6 s | 3,570 | 77.5 | 1.7 | 23.6 |

## In-the-Wild

n = 31,779 · accuracy 92.9% · real → FAKE 2.9% · fake → REAL 14.0% · EER 6.62%

**Accuracy by how many members share the final verdict**

| members agreeing | n | accuracy % |
|---|---|---|
| 1 of 3 | 1,066 | 46.8 |
| 2 of 3 | 12,187 | 87.1 |
| 3 of 3 | 18,526 | 99.4 |

**Accuracy by confidence of the merged probability**

| confidence | n | accuracy % |
|---|---|---|
| very low (50±10%) | 2,681 | 57.9 |
| low (60–70% / 30–40%) | 3,995 | 83.0 |
| medium (70–80% / 20–30%) | 6,654 | 94.6 |
| high (80–90% / 10–20%) | 8,459 | 99.3 |
| very high (≥90% / ≤10%) | 9,990 | 99.9 |

**By clip duration**

| duration | n | accuracy % | real → FAKE % | fake → REAL % |
|---|---|---|---|---|
| < 2 s | 7,084 | 89.5 | 8.3 | 16.5 |
| 2–3 s | 7,127 | 92.5 | 1.5 | 20.7 |
| 3–4 s | 5,174 | 94.8 | 1.4 | 13.3 |
| 4–6 s | 6,115 | 95.4 | 0.8 | 10.2 |
| ≥ 6 s | 6,279 | 93.4 | 0.3 | 11.4 |

**Hardest In-the-Wild speakers** (lowest accuracy, ≥ 30 clips)

| speaker | n | accuracy % |
|---|---|---|
| Frank Sinatra | 55 | 50.9 |
| Adam Driver | 217 | 54.4 |
| Nelson Mandela | 381 | 69.0 |
| Queen Elizabeth II | 464 | 76.7 |
| Milton Friedman | 589 | 78.3 |
| Robert Kardashian | 143 | 78.3 |
| Alan Watts | 378 | 80.2 |
| Jerry Seinfeld | 51 | 82.4 |
| Malcolm X | 176 | 83.0 |
| Bill Clinton | 1832 | 83.7 |
