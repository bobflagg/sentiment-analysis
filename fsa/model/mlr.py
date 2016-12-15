# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:19:05 2016

@author: birksworks
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

class MLR(BaseEstimator, ClassifierMixin):
    best_settings = {
        'ag':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 5.3182073751190799,
            'test-accuracy':0.9239
        },
        'amazon-f':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 5.3182073751190799,
            'test-accuracy':0.5802
        },
        'amazon-p':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 5.3182073751190799,
            'test-accuracy':0.9239
        },
        'dbp':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 1.0,
            'test-accuracy':0.9821
        },
        'imdb-50K':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 2.3751035233654019,
            'test-accuracy':0.8899
        },
        'rt-s':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 9.9380510738102377,
            'test-accuracy':0.7790
        },
        'stsa-f':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 1.0857064404836059,
            'test-accuracy':0.3946
        },
        'stsa-p':{
            'max_iter':100, 
            'fit_intercept': True, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 9.9445041732210306,
            'test-accuracy':0.8116
        },
        'yelp-p':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 1.0,
            'test-accuracy':0.9821
        },
        'yelp-f':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 1.0,
            'test-accuracy':0.6257
        },
        'yelp-p':{
            'max_iter':100, 
            'fit_intercept': False, 
            'use_idf': True, 
            'ngram_range': (1, 2), 
            'penalty': 'l2', 
            'C': 1.0,
            'test-accuracy':0.9530
        },
    }

    def __init__(
        self, 
        C=1.0, 
        penalty=u'l2', 
        max_iter=100, 
        fit_intercept=False,
        ngram_range=(1,1),
        use_idf=True,
        tokenizer=None,
        stop_words=None        
    ):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept 
        self.ngram_range = ngram_range 
        self.use_idf = use_idf 
        self.tokenizer = tokenizer 
        self.stop_words = stop_words 
        self.model = None
 
    def fit(self, X, y):
        self.model = Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=self.tokenizer, stop_words=self.stop_words, ngram_range=self.ngram_range)),
            ('transformer', TfidfTransformer(norm=self.penalty, use_idf=self.use_idf)),
            ('classifier', LogisticRegression(C=self.C, penalty=self.penalty, fit_intercept=self.fit_intercept, max_iter=self.max_iter)),
        ])
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)
               
