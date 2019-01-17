# encoding: utf-8

import numpy as np
from collections import Counter


class BaseNB(object):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha * 1.0
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def train(self, train_feature, train_label):
        pass

    def predict(self, test_feature):
        pass

    def score(self, y_true, y_predict):
        pass
