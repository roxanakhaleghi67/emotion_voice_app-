import librosa
import numpy as np
import tensorflow as tf


class EmotionRecognizer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return np.expand_dims(mfccs, axis=0)

    def predict_emotion(self, audio_path):
        features = self.extract_features(audio_path)
        prediction = self.model.predict(features)
        return prediction.argmax(), prediction.max()
