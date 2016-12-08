import numpy as np
import os
import pandas as pd
import pyprind

# Naive Bayes and Text Classification
# http://sebastianraschka.com/Articles/2014_naive_bayes_1.html

def prepare_imdb_data(directory='/home/data/sentiment-analysis-and-text-classification/baselines-and-bigrams/aclImdb'):
    '''
    Prepares raw IMDB movie review data for convenient use in training and evaluating
    sentiment analysis models.
    
    Args:
        directory: string
              The path to directory containing raw IMDB movie review data.
    '''
    pbar = pyprind.ProgBar(50000)
    labels = {'pos':1, 'neg':0}
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path ='%s/%s/%s' % (directory, s, l)
            for file in os.listdir(path):            
                with open(os.path.join(path, file), 'r') as infile:
                    txt = infile.read()
                    df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()
    #np.random.seed(1234)
    #df = df.reindex(np.random.permutation(df.index))
    df.columns = ['review', 'sentiment']
    df.to_csv('%s/data-frame.csv' % directory, index=False)
    
prepare_imdb_data()