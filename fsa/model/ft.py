# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:59:04 2016

@author: birksworks
"""
import fasttext
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

class FastText(BaseEstimator, ClassifierMixin):
    output = '/tmp/fast-text'
    best_settings = {
        'imdb-50K':{
            'dim': 10, 
            'lr': 0.86722412455613962, 
            'word_ngrams': 2, 
            'epoch': 10,
            'test-accuracy':0.8931
        },
        'rt-s':{
            'dim': 10, 
            'lr': 0.348758707295704912, 
            'word_ngrams': 2, 
            'epoch': 10,
            'test-accuracy':0.7618
        },
        'stsa-p':{
            'dim': 10, 
            'lr': 0.62568504214843879, 
            'word_ngrams': 2, 
            'epoch': 10,
            'test-accuracy':0.8127
        },
        'stsa-f':{
            'dim': 10, 
            'lr': 0.14001134260654835, 
            'word_ngrams': 2, 
            'epoch': 10,
            'test-accuracy':0.4199
        }
    }

    def __init__(self, 
             dim=10, 
             lr=0.5, 
             epoch=5, 
             min_count=1, 
             word_ngrams=1,
             bucket=2000000, 
             thread=4, 
             silent=1, 
             label_prefix='__label__'
    ):
        self.dim = dim
        self.lr = lr
        self.epoch = epoch
        self.min_count = min_count
        self.word_ngrams = word_ngrams
        self.bucket = bucket
        self.thread = thread
        self.silent = silent
        self.label_prefix = label_prefix
        self.model = None
        
    def store_training_data(self, X, y):        
        fname = '/tmp/fast-text-training-data-%s.csv' % os.getpid()
        temp = open(fname, 'w')
        for text, label in zip(X,y):
            temp.write("%s%s , %s\n" % (self.label_prefix, label, text))
        temp.close()
        return fname
               
    def fit(self, X, y):
        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        
        input_file = self.store_training_data(X, y)
        self.output ='/tmp/fast-text-model-%s' % os.getpid()
        self.model = fasttext.supervised(
            input_file, 
            self.output, 
            dim=self.dim, 
            lr=self.lr, 
            epoch=self.epoch, 
            min_count=self.min_count, 
            word_ngrams=self.word_ngrams, 
            bucket=self.bucket, 
            thread=self.thread, 
            silent=self.silent, 
            label_prefix=self.label_prefix
        )
        # Clean up the temporary training data file:
        os.remove(input_file)
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        #check_is_fitted(self, ['X_', 'y_'])
        labels = [int(l[0] )for l in self.model.predict(X)]
        return labels




'''
path = '/home/data/sentiment-analysis-and-text-classification/fasttext/dbpedia_csv/dbpedia.train'
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

classifier = FastText()
classifier.fit(X, y)
path = '/home/data/sentiment-analysis-and-text-classification/fasttext/dbpedia_csv/dbpedia.test'
fp = open(path, 'r')
X_test = []
y_test = []
for line in fp:
    line = line.strip()
    if line:
        parts = line.split(' , ')
        label = parts[0][9:].strip()
        text = " , ".join(parts[1:])
        X_test.append(text)
        y_test.append(label)
# Test the classifier
result = classifier.model.test(path)
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)

print('---->>>>', classifier.score(X_test, y_test))


texts = ['birchas chaim , yeshiva birchas chaim is a orthodox jewish mesivta \
        high school in lakewood township new jersey . it was founded by rabbi \
        shmuel zalmen stein in 2001 after his father rabbi chaim stein asked \
        him to open a branch of telshe yeshiva in lakewood . as of the 2009-10 \
        school year the school had an enrollment of 76 students and 6 . 6 \
        classroom teachers ( on a fte basis ) for a student–teacher ratio of \
        11 . 5 1 .'] * 100
labels =classifier.model.predict(texts)
print(labels)

labels = classifier.predict(texts)
'''
