# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:59:03 2016

@author: birksworks
"""
from fsa.model.ft import FastText
from fsa.model.mlr import MLR
from fsa.model.nbsvm import NBSVM
from fsa.util import LOGGER, tokenizer
import pandas as pd
from scipy.stats import uniform
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

DATA_DIRECTORY = '/home/code/nlp/sentiment-analysis-text-classification/sentiment-analysis/data/'
PARTITION_MAP = {
    'ag':120000,
    'imdb-50K':25000,
    'rt-s':7997,
    'stsa-f':8544,
    'stsa-p':6920
}   
MODEL_PROPERTIES_MAP = {
    'fasttext':{
        'model':FastText(),
        'distributions':{
            'dim': [10],
            'lr': uniform(0.0, 1.0),
            'epoch':[10], 
            'word_ngrams':[2]
        }
    },
    'mlr':{
        'model':MLR(),
        'distributions':{
            'penalty': [u'l1', u'l2'],
            'C': uniform(0,10),
            'max_iter':[100], 
            'fit_intercept':[False, True],
            'ngram_range':[(1,1), (1,2)],
            'use_idf':[False, True]
        }
    },
    'nbsvm':{ 
        'model':NBSVM(),
        'distributions':{
            'ngram_range': [(1,2)],
            'stop_words': [None, stop],
            'beta': uniform(0.1, 0.8),
            'C':  uniform(0.0, 5.0)
        }
    }
}   

def dump_results(search, X_test, y_test, target_names=['neg', 'pos']):
    LOGGER.info(search.best_score_)
    LOGGER.info(search.best_params_)
    predicted = search.predict(X_test)
    LOGGER.info('Test Accuracy: %.4f' % search.score(X_test, y_test))
    LOGGER.info(metrics.classification_report(y_test, predicted))
    LOGGER.info(metrics.confusion_matrix(y_test, predicted))    
    
def optimize_model(model, distributions, X_train, y_train, X_test, y_test, n_iter=20, target_names=['neg', 'pos']):
    search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=distributions, 
        n_iter=n_iter,
        scoring='accuracy',
        cv=5, 
        verbose=1,
        n_jobs=4
    )
    search.fit(X_train, y_train);
    dump_results(search, X_test, y_test, target_names=target_names)
  
class Optimizer():

    def __init__(self, mtype, dataset):
        self.mtype = mtype
        self.dataset = dataset       
        self.model = MODEL_PROPERTIES_MAP[self.mtype]['model']

    def load_data(self, dtype):
        if self.mtype == 'fasttext':
            path = '%s/%s.%s' % (DATA_DIRECTORY, self.dataset, dtype)
            fp = open(path, 'r')
            X = []
            y = []
            for line in fp:
                line = line.strip()
                if line:
                    parts = line.split(' , ')
                    label = int(parts[0][9:].strip())
                    text = " , ".join(parts[1:])
                    X.append(text)
                    y.append(label)
            fp.close()
        else:
            n = PARTITION_MAP[self.dataset]
            df = pd.read_csv('%s/%s.csv' % (DATA_DIRECTORY, self.dataset))
            #df['review'] = df['review'].apply(preprocessor)
            if dtype == 'train':
                X = df.loc[:n, 'review'].values
                y = df.loc[:n, 'sentiment'].values
            else:
                X = df.loc[n:, 'review'].values
                y = df.loc[n:, 'sentiment'].values
        return X, y
        
    def optimize(self, n_iter=100):
        distributions = MODEL_PROPERTIES_MAP[self.mtype]['distributions']
        X_train, y_train = self.load_data('train')
        LOGGER.info("Training set size = %d." % len(X_train))
        search = RandomizedSearchCV(
            estimator=self.model, 
            param_distributions=distributions, 
            n_iter=n_iter,
            scoring='accuracy',
            cv=5, 
            verbose=1,
            n_jobs=-1
        )
        search.fit(X_train, y_train);
        X_test, y_test = self.load_data('test')
        LOGGER.info("Testing set size = %d." % len(X_test))
        dump_results(search, X_test, y_test)
               
    def validate(self):
        settings = self.model.best_settings[self.dataset]
        accuracy = settings['test-accuracy']
        del settings['test-accuracy']
        self.model.set_params(**settings)



        X_train, y_train = self.load_data('train')
        LOGGER.info("Training set size = %d." % len(X_train))
        self.model.fit(X_train, y_train)
        X_test, y_test = self.load_data('test')
        LOGGER.info("Testing set size = %d." % len(X_test))
        predicted = self.model.predict(X_test)
        LOGGER.info('Test Accuracy: %.4f vs %.4f.' % (self.model.score(X_test, y_test), accuracy))
        LOGGER.info(metrics.classification_report(y_test, predicted))
        LOGGER.info(metrics.confusion_matrix(y_test, predicted))    
  
class FastTextOptimizer(Optimizer):

   def __init__(self, dataset):
        super(FastTextOptimizer, self).__init__('fasttext', dataset)
        settings = FastText.best_settings[dataset]
        self.accuracy = settings['test-accuracy']
        del settings['test-accuracy']
        self.model = FastText(**settings)
  
class MLROptimizer(Optimizer):

   def __init__(self, dataset):
        super(MLROptimizer, self).__init__('mlr', dataset)
        settings = MLR.best_settings[dataset]
        self.accuracy = settings['test-accuracy']
        del settings['test-accuracy']
        self.model = MLR(**settings)
  
class NBSVMOptimizer(Optimizer):

   def __init__(self, dataset):
        super(MLROptimizer, self).__init__('mlr', dataset)
        settings = NBSVM.best_settings[dataset]
        self.accuracy = settings['test-accuracy']
        del settings['test-accuracy']
        self.model = NBSVM(**settings)
       
import optparse
if __name__ == "__main__":     
    parser = optparse.OptionParser()
    parser.add_option("-v", help="Validate best settings.", dest='validate', action='store_true')
    parser.add_option("-m", help="Model type.", dest='mtype', action='store', type='string')
    parser.add_option("-d", help="Dataset.", dest='dataset', action='store', type='string')
    parser.add_option("-i", help="Number of iterations.", dest='n_iter', action='store', type='int', default=20)
    (opts, args) = parser.parse_args()   
    optimizer = Optimizer(opts.mtype, opts.dataset)
    '''
    if opts.mtype == 'fasttext':
        optimizer = FastTextOptimizer(opts.dataset)
    elif opts.mtype == 'mlr':
        optimizer = MLROptimizer(opts.dataset)
    '''
    if opts.validate: 
        optimizer.validate()        
    else:
        LOGGER.info("Optimizing %s on %s with %d iterations." % (opts.mtype, opts.dataset, opts.n_iter))
        optimizer.optimize(n_iter=opts.n_iter)
