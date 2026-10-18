"""Teddy, task T5 (stretch goal): LSTM on 22-day windows of features.

TODO: build sequences INSIDE segments only; scaler fit on train only; early stopping on a
chronological validation tail; average 5 seeds; time-box to 8 hours. If it does not beat
Logistic Regression on validation, say so in the report and explain why (about 5,000 daily
observations, low signal to noise).  Needs `torch` (pip install torch) - not in requirements.txt.
"""
from src.models.base import Forecaster


class LSTMForecaster(Forecaster):
    name = "lstm"

    def fit(self, train, horizon):
        raise NotImplementedError("Teddy: implement T5 (optional)")

    def predict(self, history, test):
        raise NotImplementedError
