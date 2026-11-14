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
spam_counts = {}
for word in vocabulary_list:
    ham_counts[word] = 0
    spam_counts[word] = 0

for email in training_data[:number_of_training_emails]:
    words = set(email.split())
    for word in words:
        ham_counts[word] += 1

for email in training_data[number_of_training_emails:]:
    words = set(email.split())
    for word in words:
        spam_counts[word] += 1


# print(spam_counts["account"])


# smoothing (ex "bank" occurs in spam, but has 0 in real emails (chaning to 0 to small prob.))
word_probability_ham = {}
word_probability_spam = {}
for word in vocabulary_list:
    word_probability_ham[word] = (1+ ham_counts[word]) / (number_of_training_emails + 2)
    word_probability_spam[word] = (1 + spam_counts[word]) / (number_of_training_emails+2)



# print(word_probability_ham["bank"])
# print(word_probability_spam["bank"])

prior_ham = number_of_training_emails / len(training_data) # 0.5

prior_spam = number_of_training_emails / len(training_data) # should be 0.5 as the other


predictions_list = []
for email in testing_data:
    words_in_email = set(email.split() )

    p_ham = prior_ham
    p_spam = prior_spam

    for word in words_in_email:
        if word in vocabulary_list:
            p_ham *= word_probability_ham[word]
            p_spam *= word_probability_spam[word]
        else:
            p_ham *= (0 + 1) / (number_of_training_emails + 2)
            p_spam *= (0 + 1) / (number_of_training_emails + 2)

            

    if p_ham > p_spam:
        predictions_list.append(0)
    else:
        predictions_list.append(1)

# [0] * number_of_training_emails + [1] * number_of_training_emails, 0 for non-spam (ham) and 1 for spam
testing_labels_list = [0] * (len(All_Ham) - number_of_training_emails) + [1] * (len(All_Spam) - number_of_training_emails)

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for i in range(len(testing_labels_list)):
    if testing_labels_list[i] == 1 and predictions_list[i] == 1:
        true_positive += 1
    elif testing_labels_list[i] == 0 and predictions_list[i] == 0:
        true_negative += 1
    elif testing_labels_list[i] == 0 and predictions_list[i] == 1:
        false_positive += 1
    else:
        false_negative += 1


print("True positive:", true_positive)
print("True negative:", true_negative)
print("False positive:", false_positive)
print("False negative:", false_negative)


accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
print(f"Accuracy: {accuracy:.3f}")

precision = true_positive / (true_positive + false_positive)
print(f"Precision:{ precision:.3f}")

recall = true_positive / (true_positive + false_negative)
print(f"Recall: {recall:.3f}")

f1_score = 2 * precision * recall / (precision + recall)
print(f"F1 score: {f1_score:.3f}");



