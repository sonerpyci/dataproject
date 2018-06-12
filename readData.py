# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:38:01 2018

@author: soner
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')
words = stopwords.words("english")

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
    df = pd.DataFrame(columns=["text","Ironic","subject","cleanText"])
    
    i = 0
    
    dataset = pd.read_csv(filename, sep = '\n', header=None)
    
    
    # Bu for bloğu altında düzenleme işlemleri gerçekleştiriliyor.
    for index, row in dataset.iterrows():
        columns = row.loc[i].split("\t") #Okunan karışık csv dosyasını tab ile seperate ediyor. Her bir parça bir field oluyor.
    
        # print(words[0] +'\t',words[1]+ '\t',words[-1] + '\n' ) # Dataları doğru ayırmış mıyım diye
    
        txt =  columns[1] + ' ' + columns[-1] #Fieldlardan metin içerenleri birleştiriyoruz.
        
        #Kelime sayısı feature'mız var fakat kelime sayısını metnin ilk halinden çıkarmalıyız çünkü;
        #veri düzenlendiğinde on it as gibi kelimeler atılıyor. Yanlış saymayalım.
        word_count = len(txt.split()) #oluşan text üzerinden kelime sayısını  hesaplıyoruz. Bu bizim featuremız olacak.
        vowel_count, exclamation_count, interrogation_count = count_features(txt)
        #Proje için ihtiyacımız olan kısımları filtreledik ve df verisetimize bunları ekledik.
        df = df.append({'Ironic' : columns[0].replace('"', ''), 'text' : txt, 'subject' : columns[3].lower(), 'word_count' : word_count, 'vowel_count' : vowel_count, 'exclamation_count' : exclamation_count, 'interrogation_count' : interrogation_count }, ignore_index=True)
        print(index)#Programın yürüyüp yürümediğine dair izlenimimiz olması için koyuldu.
    
    df['cleanText'] = df['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    #see,saw,seen ===> see gibi bir yapı. aynı anlamlı fakat farklı çekimlenmiş kelimeleri düzeltir.
    #aynı zamanda kelimelerin son harflerini siliyordu bunu neden yapıyor bakmak gerek ?? 
    return df
