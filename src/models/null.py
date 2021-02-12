import pandas as pd
from statistics import mode
import numpy as np

class NullModel():
    def __init__(self, target_type: str = 'regression'):
        self.target_type = target_type
        self.y = None
        self.pred_value = None
        self.preds = None

    def fit(self, y):
        self.y = y
        if self.target_type == 'regression':
            self.pred_value = y.mean()
        else:
            self.pred_value = mode(y)

    def get_length(self, y):
        return len(self.y)

    def predict(self, y):
        self.preds = [self.pred_value] * self.get_length(y)
        return self.preds

    def fit_predict(self, y):
        self.fit(y)
        return self.predict(y)