# Legacy (v1) material

These are the original Kaggle notebooks and TensorFlow weights from the first version of the project (CNN-BiLSTM on MFCCs, LJSpeech vs WaveFake-MelGAN).

They are kept for reference only. **Do not use the numbers they report.** The evaluation notebook re-splits the data with no fixed seed, so most of its "test" clips were also used in training, and the test set was used for early stopping. The reported 98.3% accuracy is therefore inflated.

The v2 pipeline (`src/adfd`, see `docs/IMPLEMENTATION_PLAN.md`) replaces them with leak-free splits and EER-based evaluation on ASVspoof 2019 LA, In-the-Wild and DEEP-VOICE.

- `Model_Training.ipynb`: original training (Kaggle TPU)
- `Model_Evaluation.ipynb`: original evaluation
- `test.ipynb`: converted the TPU-trained weights to `savedmodels/updated_model.*` (used by `app_v1_tensorflow.py`, in `legacy/savedmodels/`)
- `my_model.h5`: original weights exported from the training notebook
