# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:19:32 2016

@author: birksworks
"""

import logging
from logging.handlers import RotatingFileHandler
import re


def configure_logger(name="sentiment-analysis", path="/tmp/sentiment-analysis.log", level=logging.INFO, maxBytes=10048576, backupCount=5):
    logger = logging.getLogger(name)
    while len(logger.handlers) > 0: logger.removeHandler(logger.handlers[0])
    logger.setLevel(level)
    handler = RotatingFileHandler(path, maxBytes=maxBytes, backupCount=backupCount)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(formatter)
    logger.setLevel(level)
    return logger
    
LOGGER = configure_logger()

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

def tokenizer(text): 
    text = preprocessor(text)
    return text.split()
