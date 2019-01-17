# encoding: utf-8

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from navie_bayes.MultinomialNB import MultinomialNB
from navie_bayes.BernoulliNB import BernoulliNB
from navie_bayes.GaussianNB import GaussianNB

def get_feature_bow(file_path="./data/sms"):
    df = pd.read_table(file_path, header=None, sep='\t')
    label, context = df[0].values, df[1].values
    train_context, test_context, train_label, test_label = train_test_split(context, label, test_size=0.2)

    Vectorizer = CountVectorizer(max_features=2000,
                                 analyzer=str.split)
    Vectorizer = Vectorizer.fit(train_context)
    train_feature = Vectorizer.transform(train_context).toarray()
    test_feature = Vectorizer.transform(test_context).toarray()
    return train_feature, train_label, test_feature, test_label


def get_feature_tf_idf(file_path="./data/sms"):
    df = pd.read_table(file_path, header=None, sep='\t')
    label, context = df[0].values, df[1].values
    train_context, test_context, train_label, test_label = train_test_split(context, label, test_size=0.2)

    Vectorizer = TfidfVectorizer(max_features=2000,
                                 analyzer=str.split)
    Vectorizer = Vectorizer.fit(train_context)
    train_feature = Vectorizer.transform(train_context).toarray()
    test_feature = Vectorizer.transform(test_context).toarray()
    return train_feature, train_label, test_feature, test_label


if __name__ == '__main__':
    # train_feature, train_label, test_feature, test_label = get_feature_bow()
    # navieBayes = MultinomialNB()
    # navieBayes.train(train_feature, train_label)
    # print(navieBayes.score(test_feature, test_label))

    train_feature, train_label, test_feature, test_label = get_feature_tf_idf()
    navieBayes = GaussianNB()
    navieBayes.train(train_feature, train_label)
    print(navieBayes.score(test_feature, test_label))

