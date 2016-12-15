# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:56:32 2016

@author: birksworks

Rolling your own estimator
http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator

Creating your own estimator in scikit-learn
http://danielhnyk.cz/creating-your-own-estimator-scikit-learn
"""

from fsa.util import tokenizer
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.pipeline import Pipeline
from scipy.sparse import spmatrix, coo_matrix
from sklearn.svm import LinearSVC
import string
class NBSVM(BaseEstimator, ClassifierMixin):
    best_settings = {
        'ag':{
            'C': 1.0, 
            #'beta': 0.17255005150558037, 
            'beta': 0.25, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.910 # 0.9170
            #'test-accuracy':0.9096
        },
        'amazon-f':{
            'C': 1.0, 
            #'beta': 0.17255005150558037, 
            'beta': 0.25, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.910 # 0.9170
            #'test-accuracy':0.9096
        },
        'amazon-p':{
            'C': 1.0, 
            #'beta': 0.17255005150558037, 
            'beta': 0.25, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.910 # 0.9170
            #'test-accuracy':0.9096
        },
        'dbp':{
            'C': 1.0, 
            #'beta': 0.17255005150558037, 
            'beta': 0.25, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.910 # 0.9170
            #'test-accuracy':0.9096
        },
        'imdb-50K':{
            'C': 1.0, 
            #'beta': 0.17255005150558037, 
            'beta': 0.25, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.910 # 0.9170
            #'test-accuracy':0.9096
        },
        'imdb-50K':{
            'C': 1.0, 
            #'beta': 0.17255005150558037, 
            'beta': 0.25, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.910 # 0.9170
            #'test-accuracy':0.9096
        },
        'rt-s':{
            'C': 1.0, 
            'beta': 0.13919541427610971, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.7921
        },
        'stsa-p':{
            'C': 4.8262879748265011, 
            'beta': 0.16755660997066729, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.8276
        },
        'stsa-f':{
            'C': 0.1447984352097792, 
            'beta': 0.16215593797254996, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.4186
        },
        'yelp-f':{
            'C': 1.0, 
            #'beta': 0.17255005150558037, 
            'beta': 0.25, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.910 # 0.9170
            #'test-accuracy':0.9096
        },
        'yelp-p':{
            'C': 1.0, 
            #'beta': 0.17255005150558037, 
            'beta': 0.25, 
            'ngram_range': (1, 2), 
            'stop_words': None,
            'test-accuracy':0.910 # 0.9170
            #'test-accuracy':0.9096
        }
    }
    
    def __init__(self, 
        ngram_range=(1,1), 
        stop_words=None, 
        tokenizer=tokenizer,
        alpha=1, 
        C=1, 
        beta=0.25, 
        fit_intercept=False
    ):
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.model = None
               
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        token_pattern = r'\w+|[%s]' % string.punctuation

        self.model = Pipeline([
            ('vectorizer', CountVectorizer(
                ngram_range=self.ngram_range, 
                stop_words=self.stop_words, 
                #tokenizer=self.tokenizer,
                token_pattern=token_pattern,
                binary=True
            )),
            ('classifier', Classifier(alpha=self.alpha, C=self.C, beta=self.beta, fit_intercept=self.fit_intercept))
        ])
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class Classifier(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):

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
