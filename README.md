# Emotion Detection from Voice

Predicts the emotion in a short speech clip (for example angry, happy or neutral) from MFCC
features and a trained TensorFlow/Keras model.

## What it does

`emotion_voice_app.py` defines `EmotionRecognizer`:

1. loads the first 3 seconds of the clip (from 0.5 s) with `librosa`;
2. averages 40 MFCC coefficients into one feature vector;
3. runs the model and returns the predicted class index and its confidence.

```python
from emotion_voice_app import EmotionRecognizer

model = EmotionRecognizer("my_model.h5")
label, confidence = model.predict_emotion("sample.wav")
```

## Requirements

- Python 3.9+
- `tensorflow`, `librosa`, `numpy` (see `requirements.txt`)

```bash
python -m pip install -r requirements.txt
```

## Notes and limitations

- A trained model file is **not** included; you must supply one trained on your own dataset.
- The class index means nothing without the label list used during training — keep that list
  with the model.
- The model expects the same 40-MFCC feature layout used here; changing `n_mfcc` breaks it.

## Status

Single-module prototype. Planned work: label mapping, a training script, and a small
command-line interface.

## License

No license file is present yet. Until one is added, the default copyright applies.
