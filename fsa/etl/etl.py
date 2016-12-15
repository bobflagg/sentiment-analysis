# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:08:34 2016

@author: birksworks
"""

import os
import pandas as pd
import pyprind

SOURCE_DIRECTORY = '/home/code/nlp/sentiment-analysis-text-classification/text_convnet/data'
TARGET_DIRECTORY = '/home/code/nlp/sentiment-analysis-text-classification/sentiment-analysis/data'

def process_file(path, df, pbar):
    fp = open(path)
    for line in fp:
        line = line.strip()
        if line:
            label = line[0]
            text = line[2:]
            df = df.append([[text, label]], ignore_index=True)
            pbar.update()
    fp.close()
    return df
    
def process_dataset(dtype='stsa.binary'):  
    df = pd.DataFrame()
    path = os.path.join(SOURCE_DIRECTORY, '%s.train' % dtype)  
    df = process_file(path, df)
    path = os.path.join(SOURCE_DIRECTORY, '%s.test' % dtype)  
    df = process_file(path, df)
    df.columns = ['review', 'sentiment']  
    if dtype=='stsa.binary': dtype = 'p'
    else: dtype = 'f'
    df.to_csv('/home/code/nlp/sentiment-analysis-text-classification/sentiment-analysis/data/stsa-%s.csv' % dtype, index=False)

PARTITION_MAP = {
    'imdb-50K':25000,
    'rt-s':7997,
    'stsa-p':6920,
    'stsa-f':8544
}   

def split_data(dataset='stsa-p'):
    n = PARTITION_MAP[dataset]
    df = pd.read_csv('/home/code/nlp/sentiment-analysis-text-classification/sentiment-analysis/data/%s.csv' % dataset)
    X_train = df.loc[:n, 'review'].values
    y_train = df.loc[:n, 'sentiment'].values
    X_test = df.loc[n:, 'review'].values
    y_test = df.loc[n:, 'sentiment'].values    
    fp = open('/home/code/nlp/sentiment-analysis-text-classification/sentiment-analysis/data/%s-train.csv' % dataset, 'w')
    for i in range(X_train.shape[0]):
        text = X_train[i]
        label = y_train[i]
        fp.write("%s , %s\n" % (label, text))
    fp.close()
    fp = open('/home/code/nlp/sentiment-analysis-text-classification/sentiment-analysis/data/%s-test.csv' % dataset, 'w')
    for i in range(X_test.shape[0]):
        text = X_test[i]
        label = y_test[i]
        fp.write("%s , %s\n" % (label, text))
    fp.close()

import csv

def process_test_train_file(path, writer, pbar):
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if line:
            parts = line.split(' , ')
            label = int(parts[0][9:].strip())
            text = " , ".join(parts[1:])
            writer.writerow([text, label])
            pbar.update()
    fp.close()

def from_test_train(dataset, ntest, ntrain):    
    ofp = open(os.path.join(TARGET_DIRECTORY, '%s.csv' % dataset), "w")
    writer = csv.writer(ofp)
    writer.writerow(['review','sentiment'])

    path = os.path.join(TARGET_DIRECTORY, '%s.train' % dataset)  
    process_test_train_file(path, writer, pbar=pyprind.ProgBar(ntrain))
     
    path = os.path.join(TARGET_DIRECTORY, '%s.test' % dataset)  
    process_test_train_file(path, writer, pbar=pyprind.ProgBar(ntest))
    ofp.close()

#process_dataset(dtype='stsa.binary')
#process_dataset(dtype='stsa.fine')
#split_data(dataset)
import optparse
if __name__ == "__main__":     
    parser = optparse.OptionParser()
    parser.add_option("-c", help="Prepare CSV file.", dest='csv', action='store_true')
    parser.add_option("-d", help="Dataset.", dest='dataset', action='store', type='string')
    parser.add_option("--test-size", help="Number of testing records", dest='ntest', action='store', type='int')
    parser.add_option("--train-size", help="Number of training records", dest='ntrain', action='store', type='int')
    (opts, args) = parser.parse_args()   
    if opts.csv: from_test_train(opts.dataset, opts.ntest, opts.ntrain)
