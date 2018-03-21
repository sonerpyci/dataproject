# -*- coding: utf-8 -*-
"""
This is a temporary script file.
"""

import csv
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import pandas as pd

dataset = pd.read_csv("deneme.csv", sep = '\n', header=None)
df = pd.DataFrame(columns=["Text","Ironic"])

i = 0

for index, row in dataset.iterrows():
    words = row.loc[i].split("\t")

    # print(words[0] +'\t',words[1]+ '\t',words[-1] + '\n' ) # Dataları doğru ayırmış mıyım diye

    txt =  words[1] + ' ' + words[-1]
    ss = 'a string with "double" quotes'
    txt = txt.replace('"', '')
    newRow = [words[0], txt] # { 'ironik mi?' , 'Text' }  Sadece görsel olarak formata bakmak için eklenen bir satır.
    
    df = df.append({'Ironic' : words[0], 'Text' : txt}, ignore_index=True)
    #print(df['Ironic'].to_string(index = False))
    print(index)

df.to_csv(path_or_buf ="C:/Users/soner/Desktop/temp.csv", sep = ',', header = None, index=False)

with open('C:/Users/soner/Desktop/temp.csv', 'r') as csvfile:

    cl = NaiveBayesClassifier(csvfile, format="csv")

print("Sonuc : " + cl.classify("I feel amazing!"))
print("Sonuc : " + cl.classify("Are you sick!"))
