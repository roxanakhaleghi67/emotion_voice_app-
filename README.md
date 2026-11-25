# Emotion Detection from Voice

Predict emotions from short audio clips using a trained TensorFlow model and MFCC audio features.

## Features

- Loads a trained Keras/TensorFlow model (`.h5`) from disk
- Extracts 40 MFCC features from an audio clip with `librosa`
- Analyzes a fixed 3-second window of the input audio
- Returns the predicted emotion class index and its confidence score

## Requirements

- Python 3.8 or newer
- `tensorflow` (2.x)
- `librosa`
- `numpy`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from emotion_voice_app import EmotionRecognizer

model = EmotionRecognizer("my_model.h5")
label, confidence = model.predict_emotion("sample.wav")
print(label, confidence)
```

> **Note:** A trained model file is **not** included in this repository. You must supply your own `EmotionRecognizer`-compatible model (for example, trained on a dataset such as RAVDESS) before predictions can be made. The class index maps to whatever label order your model was trained with.

## Project Structure

```text
emotion_voice_app-/
├── emotion_voice_app.py   # EmotionRecognizer implementation
├── requirements.txt
└── README.md
```

## License

This project is released under the **MIT License**.

## Author

**Roxana Khaleghi**
