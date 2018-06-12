# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:20:34 2018

@author: soner
"""

import pandas as pd



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



def test_model(model):
    inpt = input("Please enter ur test Data With format : [ subject || text ] \n")
    
    subject = inpt.split('||')[0]
    text = inpt.split('||')[1]
    word_count = len(inpt.split('||')[1].split())
    vowel_count, exclamation_count, interrogation_count = count_features(text)
    
    dct = {
        "subject": subject,
        "text": text,
        "word_count": word_count,
        "vowel_count": vowel_count,
        "exclamation_count": exclamation_count,
        "interrogation_count": interrogation_count
    }
    
    test_df = pd.DataFrame(dct, index=[0])

   
    print(test_df.drop(columns=['text']))
    return model.predict(test_df.drop(columns=['text']))