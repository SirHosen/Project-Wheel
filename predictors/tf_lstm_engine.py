# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import logging

# Ensure TensorFlow logging is minimized
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from .base import BasePredictor
from config import settings

class TfLstmEngine(BasePredictor):
    """LSTM-based predictor for sequence learning."""
    
    def __init__(self, sequence_length=settings.LSTM_SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.valid_numbers = settings.VALID_NUMBERS
        self.num_classes = len(self.valid_numbers)
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.valid_numbers)
        
        self.model = self._build_model()
        self.history_metrics = {'loss': [], 'accuracy': []}
        
    def _build_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(32),
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model
        
    def prepare_data(self, sequence_list):
        if len(sequence_list) <= self.sequence_length:
            return np.array([]), np.array([])
            
        df = pd.DataFrame({'number': sequence_list})
        try:
            df['encoded'] = self.label_encoder.transform(df['number'])
        except Exception as e:
            logging.warning(f"Label encoding failed in prepare_data: {e}. Sequence contains invalid numbers.")
            return None, None
            
        windows = [df['encoded'].iloc[i:i+self.sequence_length].values 
                  for i in range(len(df) - self.sequence_length)]
        targets = df['encoded'].iloc[self.sequence_length:].values
        
        X = np.array(windows).reshape(-1, self.sequence_length, 1)
        X = X / (self.num_classes - 1)
        y = np.array(targets)
        
        return X, y
        
    def train(self, history_list, epochs=1):
        """Train the model incrementally or in bulk."""
        X, y = self.prepare_data(history_list)
        if X is None or y is None:
            return False
        if len(X) > 0:
            hist = self.model.fit(X, y, epochs=epochs, verbose=0)
            self.history_metrics['loss'].extend(hist.history['loss'])
            self.history_metrics['accuracy'].extend(hist.history['accuracy'])
            return True
        return False

    def predict_next(self, history: list) -> list:
        if len(history) < self.sequence_length:
            # Fallback to equal probability
            prob = 1.0 / self.num_classes
            return [{"number": n, "confidence": prob} for n in self.valid_numbers]
            
        seq = history[-self.sequence_length:]
        df = pd.DataFrame({'number': seq})
        try:
            df['encoded'] = self.label_encoder.transform(df['number'])
        except Exception as e:
            logging.warning(f"Label encoding failed in predict_next: {e}. Sequence contains invalid numbers.")
            return None
            
        X = df['encoded'].values.reshape(1, self.sequence_length, 1)
        X = X / (self.num_classes - 1)
        
        probs = self.model.predict(X, verbose=0)[0]
        
        predictions = []
        for class_idx, prob in enumerate(probs):
            number = int(self.label_encoder.inverse_transform([class_idx])[0])
            predictions.append({"number": number, "confidence": float(prob)})
            
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions
