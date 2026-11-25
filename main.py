"""Example usage for the emotion recognizer.

Run:  python main.py path/to/model.h5 path/to/clip.wav
"""

import sys

from emotion_voice_app.model import EmotionRecognizer


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python main.py <model.h5> <audio.wav>")
    model_path, audio_path = sys.argv[1], sys.argv[2]
    recognizer = EmotionRecognizer(model_path)
    label, confidence = recognizer.predict_emotion(audio_path)
    print(f"emotion index: {label} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()
