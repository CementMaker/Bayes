# encoding: utf-8
import numpy as np

from navie_bayes import BaseNB
from collections import Counter


class BernoulliNB(BaseNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super(BernoulliNB, self).__init__(alpha, fit_prior, class_prior)

    def _cal_prior(self):
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
        self.piror = self._cal_prior()

        # 样本归类
        label_dict = dict()
        for feature, label in zip(self._bayes_train_feature, self._bayes_train_label):
            if label not in label_dict:
                label_dict[label] = [feature]
            else:
                label_dict[label].append(feature)

        # 特征概率计算
        self.label_feature_proba = {}
        for key in label_dict:
            self.label_feature_proba[key] = np.sum(label_dict[key], axis=0) + self.alpha
            self.label_feature_proba[key] = self.label_feature_proba[key] / (len(label_dict[key]) + self.alpha * len(label_dict[key]))

    def predict(self, test_feature):
        test_feature = np.array(test_feature)

        predict = []
        ans_dict = dict()
        for feature in test_feature:
            for label in self.label_feature_proba:
                ans_dict[label] = self.prior[label]
                for num, proba in zip(feature, self.label_feature_proba[label]):
                    ans_dict[label] = ans_dict[label] * pow(proba, num)

            proba, label_predict = 0.0, None
            for label in ans_dict:
                if ans_dict[label] >= proba:
                    label_predict = label
                    proba = ans_dict[label]
            predict.append(label_predict)
        return predict

    def score(self, test_feature, test_label):
        return np.mean(self.predict(test_feature) == test_label)