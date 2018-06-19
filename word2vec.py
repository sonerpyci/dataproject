# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 21:57:34 2018

@author: Dev
"""

from __future__ import print_function
import gensim 
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
#model = gensim.models.Doc2Vec.load('saved_doc2vec_model')  

def get_word2vec_model():
    model = KeyedVectors.load_word2vec_format('wiki.vec')
    index2word_set = set(model.wv.index2word)
    return model,index2word_set



def avg_feature_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    avg_tmp = pd.DataFrame()
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
            
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
        
    avg_tmp = avg_tmp.append(pd.Series(featureVec), ignore_index=True)
    return avg_tmp

std_vecs = pd.DataFrame()

def std_feature_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    std_vecs = pd.DataFrame()
    std_tmp = pd.DataFrame()


    for word in words:
        featureVec = np.zeros((num_features,), dtype="float32")
        if word in index2word_set:
            featureVec = np.add(featureVec, model[word]) # kelimelerin her birine ait vektör 
            std_vecs = std_vecs.append(pd.Series(featureVec), ignore_index=True)#cümleye ait vektör. 

    std_tmp = std_vecs.std()

    return std_tmp





"""
sentence_1 = "this is sentence"
sentence_1_avg_vector = avg_feature_vector(sentence_1.split(), model=word2vec_model, num_features=300,index2word_set=set(word2vec_model.wv.index2word))
print(sentence_1_avg_vector)
print("---------\n")
#get average vector for sentence 2  
sentence_2 = "this is sentence sajkdhasjh3"
sentence_2_avg_vector = avg_feature_vector(sentence_2.split(), model=word2vec_model, num_features=300,index2word_set=set(word2vec_model.wv.index2word))
print(sentence_2_avg_vector)"""