# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:06:46 2017

@author: xfang13
"""
from os import walk
import numpy as np

seed = 12345
number_of_training_emails = 500

ham_dir = 'Ham'
All_Ham = []
for (dirpath, dirnames, filenames) in walk(ham_dir):
    if filenames != []:
        for ham in filenames:
            with open(dirpath+'\\'+ham) as f:
                #read the file as a big string
                All_Ham.append(f.read())
                
spam_dir = 'Spam'
All_Spam = []
for (dirpath, dirnames, filenames) in walk(spam_dir):
    if filenames != []:
        for spam in filenames:
            with open(dirpath+'\\'+spam,encoding='latin-1') as f:
                #read the file as a big string
                All_Spam.append(f.read())           
                

randomState = np.random.RandomState(seed)
randomState.shuffle(All_Ham)
randomState.shuffle(All_Spam)

training_data = All_Ham[:number_of_training_emails] + All_Spam[:number_of_training_emails]
testing_data = All_Ham[number_of_training_emails:] + All_Spam[number_of_training_emails:] 

#important: use .split() to separate the words 
vocabulary_list = set()
for email in training_data: 
    words_in_email = email.split()
    vocabulary_list.update(words_in_email)


ham_counts = {}
for word in vocabulary_list:
    ham_counts[word] = 0

spam_counts = {}
for word in vocabulary_list:
    spam_counts[word] = 0

for email in training_data[:number_of_training_emails]:
    words = set(email.split())
    for word in words:
        ham_counts[word] += 1

for email in training_data[number_of_training_emails:]:
    words = set(email.split())
    for word in words:
        spam_counts[word] += 1


print(spam_counts["account"])

