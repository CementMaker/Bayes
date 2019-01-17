# encoding: utf-8

import math
import numpy as np
import scipy.stats, scipy

from navie_bayes import BaseNB
from collections import Counter


class GaussianNB(BaseNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super(GaussianNB, self).__init__(alpha, fit_prior, class_prior)

    def _cal_prior(self):
        print(self._bayes_train_feature.shape)
        sample_number = self._bayes_train_feature.shape[0]
        number_per_label = Counter(self._bayes_train_label).items()
        if self.fit_prior:
            self.prior = dict([(label, num * 1.0 / sample_number) for label, num in number_per_label])
        elif self.class_prior is not None:
            self.piror = dict(self.class_prior)
        else:
            self.prior = dict([(label, 1.0 / self.number_label) for label, _ in number_per_label])
        return self.prior

    def train(self, train_feature, train_label):
        self._bayes_train_feature = np.array(train_feature)
        self._bayes_train_label = np.array(train_label)
        self.number_label = len(set(self._bayes_train_label))
        self.prior = self._cal_prior()

        # 样本归类
        label_dict = dict()
        for feature, label in zip(self._bayes_train_feature, self._bayes_train_label):
            if label not in label_dict:
                label_dict[label] = [feature]
            else:
                label_dict[label].append(feature)

        self.label_feature_proba = {}
        for label in label_dict:
            label_dict[label] = np.array(label_dict[label])
            mean = np.mean(label_dict[label], axis=0)
            std = np.std(label_dict[label], axis=0)
            Gaussian = zip(mean, std)
            self.label_feature_proba[label] = np.array(list(Gaussian))

    def cal_Gaussian(self, mean, std, x):
        if std == 0:
            return 0.1
        tmp = 1.0 / math.sqrt(2 * np.pi) / std
        return tmp * np.exp(-1 * (x - mean) * (x - mean) / (2 * std * std))

    def predict(self, test_feature):
        test_feature = np.array(test_feature)

        predict = []
        ans_dict = dict()
        for feature in test_feature:
            for label in self.label_feature_proba:
                ans_dict[label] = self.prior[label]
                for num, val in zip(feature, self.label_feature_proba[label]):
                    ans_dict[label] = ans_dict[label] * self.cal_Gaussian(val[0], val[1], num)

            proba, label_predict = 0.0, None
            for label in ans_dict:
                if ans_dict[label] >= proba:
                    label_predict = label
                    proba = ans_dict[label]
            predict.append(label_predict)
        print(predict)
        return predict

    def score(self, test_feature, test_label):
        return np.mean(self.predict(test_feature) == test_label)