# P3.5 Model size and inference speed

Seed-42 checkpoints, 4 s clips. GPU: RTX 3050 Laptop (4 GB), batch 32, fp16. CPU: batch 1, fp32, 10 threads. Speed = seconds of audio scored per second.

| Model | parameters | trainable | checkpoint (MB) | GPU speed (× real time) | CPU speed (× real time) | CPU ms per clip |
|---|---|---|---|---|---|---|
| E1 CNN-BiLSTM, MFCC (v1) | 786,625 | 786,625 | 3.5 | 23,278 | 815.7 | 4.9 |
| E4a CNN-BiLSTM, LFCC + RawBoost | 786,625 | 786,625 | 3.1 | 8,579 | 288.4 | 13.9 |
| E5a LCNN, LFCC | 269,729 | 269,729 | 1.1 | 8,917 | 521.0 | 7.7 |
| E5b RawNet2 | 17,620,385 | 17,620,385 | 67.3 | 1,870 | 61.8 | 64.7 |
| E6b SE-res CNN-BiLSTM-Att | 1,422,206 | 1,422,206 | 5.5 | 4,925 | 290.8 | 13.8 |
| E7a WavLM-Base+ + CNN-BiLSTM-Att | 96,199,035 | 91,998,587 | 367.1 | 337 | 16.4 | 244.6 |
