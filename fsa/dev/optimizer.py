# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 08:31:38 2016

@author: birksworks
"""

import codecs
import json
import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from nltk.stem.porter import PorterStemmer
import re

import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from scipy.sparse import spmatrix, coo_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.svm import LinearSVC

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text
    
def tokenizer(text): 
    text = preprocessor(text)
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    text = preprocessor(text)
    return [porter.stem(word) for word in text.split()]


class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):

    def __init__(self, alpha=1, C=1, beta=0.25, fit_intercept=False):
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            coef_, intercept_ = self._fit_binary(X, y)
            self.coef_ = coef_
            self.intercept_ = intercept_
        else:
            coef_, intercept_ = zip(*[
                self._fit_binary(X, y == class_)
                for class_ in self.classes_
            ])
            self.coef_ = np.concatenate(coef_)
            self.intercept_ = np.array(intercept_).flatten()
        return self

    def _fit_binary(self, X, y):
        p = np.asarray(self.alpha + X[y == 1].sum(axis=0)).flatten()
        q = np.asarray(self.alpha + X[y == 0].sum(axis=0)).flatten()
        r = np.log(p/np.abs(p).sum()) - np.log(q/np.abs(q).sum())
        b = np.log((y == 1).sum()) - np.log((y == 0).sum())

        if isinstance(X, spmatrix):
            indices = np.arange(len(r))
            r_sparse = coo_matrix(
                (r, (indices, indices)),
                shape=(len(r), len(r))
            )
            X_scaled = X * r_sparse
        else:
            X_scaled = X * r

        lsvc = LinearSVC(
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=10000
        ).fit(X_scaled, y)

        mean_mag =  np.abs(lsvc.coef_).mean()
        coef_ = (1 - self.beta) * mean_mag * r + self.beta * (r * lsvc.coef_)
        intercept_ = (1 - self.beta) * mean_mag * b + self.beta * lsvc.intercept_

        return coef_, intercept_
        
class Optimizer():

    def __init__(self): pass

    def load_data(self):
        df = pd.read_csv('../../data/imdb-50K.csv')
        df['review'] = df['review'].apply(preprocessor)
        X_train = df.loc[:25000, 'review'].values
        y_train = df.loc[:25000, 'sentiment'].values
        X_test = df.loc[25000:, 'review'].values
        y_test = df.loc[25000:, 'sentiment'].values
        return X_train, y_train, X_test, y_test
        
    def build_model(self):
        model = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', NBSVM())
        ])
        return model
        
    def optimize(self, n_iter=100):
        X_train, y_train, X_test, y_test = self.load_data()
        model = self.build_model()
        param_distributions = {
            'vect__ngram_range': [(1,2)],
            'vect__stop_words': [None],
            'vect__tokenizer': [tokenizer],
            'clf__beta': uniform(0.1, 0.7),
            'clf__C': [1.0]
        }
        search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_distributions, 
            n_iter=n_iter,
            scoring='accuracy',
            cv=5, 
            verbose=1,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        print(search.best_score_)
        print(search.best_params_)
        predicted = search.predict(X_test)
        print('Test Accuracy: %.4f' % search.score(X_test, y_test))
        print(metrics.classification_report(y_test, predicted, target_names=['neg', 'pos']))
        print(metrics.confusion_matrix(y_test, predicted))
        
optimizer = Optimizer()
optimizer.optimize(n_iter=40)
