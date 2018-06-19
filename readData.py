# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:38:01 2018

@author: soner
"""

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import sys
sys.path.append('home/sonerpyci/Desktop/textMiner/main')

import word2vec
import TermMatrix
stemmer = SnowballStemmer('english')
words = stopwords.words("english")


vec_model,index2word_set = word2vec.get_word2vec_model()


def count_features(string):
    num_vowels=0
    num_exclamation=0
    num_interrogation=0
    for char in string:
        if char in "aeiouAEIOU":
           num_vowels = num_vowels+1
        if char is "!":
           num_exclamation = num_exclamation+1
        if char is "?":
           num_interrogation = num_interrogation+1
    return num_vowels,num_exclamation,num_interrogation


#Bu fonksiyon kaynak dosyadan veri setini okur. Okunan veri seti ile ilgili temel düzenlemeleri yapar.
def read_csv_file(filename):
    #düzenli haldeki veri seti için boş bir sample açıyoruz.
    std_df = pd.DataFrame(dtype='float32')
    avg_df = pd.DataFrame(dtype='float32')
    vect_df = pd.DataFrame(dtype='float32')
    newlst= []
    
    df = pd.DataFrame(columns=["text","Ironic","subject","cleanText"])
    
    i = 0
    
    dataset = pd.read_csv(filename, sep = '\r\n', header=None, error_bad_lines=False)
    
    
    # Bu for bloğu altında düzenleme işlemleri gerçekleştiriliyor.
    for index, row in dataset.iterrows():
        print(index)
        columns = row.loc[i].split("\t") #Okunan karışık csv dosyasını tab ile seperate ediyor. Her bir parça bir field oluyor.
    
        # print(words[0] +'\t',words[1]+ '\t',words[-1] + '\n' ) # Dataları doğru ayırmış mıyım diye
    
        txt =  columns[1] + ' ' + columns[-1] #Fieldlardan metin içerenleri birleştiriyoruz.
        txt = txt.translate(string.punctuation)
        newlst.append(txt)
        
        std_vector = word2vec.std_feature_vector(txt, model=vec_model, num_features=300,index2word_set=index2word_set)
        std_df = std_df.append(std_vector, ignore_index=True)

        avg_vector = word2vec.avg_feature_vector(txt, model=vec_model, num_features=300,index2word_set=index2word_set)
        avg_df = avg_df.append(avg_vector, ignore_index=True)
        
        
        word_count = len(txt.split()) #oluşan text üzerinden kelime sayısını  hesaplıyoruz. Bu bizim featuremız olacak.
        vowel_count, exclamation_count, interrogation_count = count_features(txt)
        #Proje için ihtiyacımız olan kısımları filtreledik ve df verisetimize bunları ekledik.
        df = df.append({'Ironic' : columns[0].replace('"', ''), 'text' : txt, 'subject' : columns[3].lower(), 'word_count' : word_count, 'vowel_count' : vowel_count, 'exclamation_count' : exclamation_count, 'interrogation_count' : interrogation_count }, ignore_index=True)
        #print(index)#Programın yürüyüp yürümediğine dair izlenimimiz olması için koyuldu.? 
    vect_df = TermMatrix.vectorizer(newlst)
    df = pd.concat([df, std_df, avg_df, vect_df], axis=1, join_axes=[df.index])     
    return df
df.to_csv("datason2.csv", index=False)
df = read_csv_file("kucuk.csv")


