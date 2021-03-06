"""
Created on Sat Jun  9 22:30:39 2018

@author: Dev_Ozan
"""
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import sys
sys.path.append("C:\Python35\Lib\site-packages")
#nltk.download()

stemmer = SnowballStemmer('english')
words = stopwords.words("english")

i=0

def vectorizer(newlst):
    vec = CountVectorizer()
    X = vec.fit_transform(newlst)
    
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) #frekans tablosu 

    #if(index % 200 == 0):
    #    print(df)
    binaryX = X.toarray()
    for i in range(len(binaryX)):
        for j in range(len(binaryX[i])):
            if(binaryX[i][j] != 0):
                binaryX[i][j] = 1
    df2 = pd.DataFrame(binaryX, columns=vec.get_feature_names())#binary tablo

    vec =  CountVectorizer(analyzer='char',ngram_range=(3,3))
    X = vec.fit_transform(newlst)
    df3 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

    binaryX = X.toarray()
    for i in range(len(binaryX)):
        for j in range(len(binaryX[i])):
            if(binaryX[i][j] != 0):
                binaryX[i][j] = 1
    df4 = pd.DataFrame(binaryX, columns=vec.get_feature_names())#binary tablo

    #print(df2)
    #print(df3)
    df = pd.concat([df, df2,df3,df4], axis=1, join_axes=[df.index])
    
    return df