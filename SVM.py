#!/usr/bin/env python
# coding: utf-8

# In[137]:


import pandas as pd
import numpy as np
import operator
import math
import random
import glob
from sklearn.svm import SVC

#read dataset
csvFile = pd.read_csv('email_dataset.csv', names = ["text", "label_num"])

emails = csvFile.to_numpy()
labels = emails[:,-1]
emails = emails[:,0]
#list of all emails
emails = emails[1:]
#labels of all emails
labels = labels[1:]

#index of all spam emails
spamEmailIndex = []
#index of all nonspam emails
nonspamEmailIndex = []
emailIndex = []
for i in range(len(labels)):
    emailIndex.append(i)
    if(labels[i] == '1'):
        spamEmailIndex.append(i)
    else:
        nonspamEmailIndex.append(i)
        
#divide dataset into 80% and 20%
spamEmailTest = random.sample(spamEmailIndex, int(0.2*len(spamEmailIndex)))
nonspamEmailTest = random.sample(nonspamEmailIndex, int(0.2*len(nonspamEmailIndex)))
#test dataset
testEmail = spamEmailTest
testEmail.extend(nonspamEmailTest)
#train dataset
trainEmail = list(set(emailIndex) - set(testEmail))
trainEmailLabels = []
for i in trainEmail:
    trainEmailLabels.append(labels[i])
    
    
#special character which is to be removed from the email
special_characters=['~','`','!','@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']',':',';','"','|','\',','?','<','>',',','.','/','0','1','2','3','4','5','6','7','8','9',"'","Subject","subject"]
for i in trainEmail:
    for sc in special_characters:
        #replace special character with empty string
        emails[i] = emails[i].replace(sc,'')
        
wordFrequency = {}
emailsAfterSplitting = len(trainEmail)*[0]
for i in range(len(trainEmail)):
    words = emails[trainEmail[i]].split()
    for j in range(len(words)):
        words[j] = words[j].strip().lower()
    emailsAfterSplitting[i] = words
    #count frequency of each word
    for word in words:
        if word in wordFrequency:
            wordFrequency[word] += 1
        else:
            wordFrequency[word] = 1

            
#sort dictionary of words in descending order
sortedWordFrequency = dict( sorted(wordFrequency.items(), key=operator.itemgetter(1),reverse=True))
#pick top 5000 words
mostFrequentWords = list(sortedWordFrequency.keys())[0:5000]
#number of times email i contains word j
doesEmailContainWord = np.zeros(shape=(len(trainEmail),len(mostFrequentWords)))

for i in range(len(trainEmail)):
    for j in range(len(mostFrequentWords)):
        if mostFrequentWords[j] in emailsAfterSplitting[i]:
            doesEmailContainWord[i][j] += 1
            
#train the svm model
clf = SVC(kernel='linear')  
clf.fit(doesEmailContainWord, trainEmailLabels) 


wrongPredictionCount = 0
special_characters=['~','`','!','@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']',':',';','"','|','\',','?','<','>',',','.','/','0','1','2','3','4','5','6','7','8','9',"'","Subject","subject"]
for i in testEmail:
    data = emails[i]
    #remove special characters
    for sc in special_characters:
        data = data.replace(sc,'')
    words = data.split()
    #remove space from word and convert string to lowercase
    for j in range(len(words)):
        words[j] = words[j].strip().lower()
        
    #check which all words of top k words are present in the current mail
    isWordPresent = len(mostFrequentWords)*[0]
    for j in range(len(mostFrequentWords)):
        if mostFrequentWords[j] in words:
            isWordPresent[j] += 1
    if(clf.predict([isWordPresent])[0] == '1'):
        #if it is predicting nonspam email as spam
        if(labels[i] == '0'):
            wrongPredictionCount += 1
    else:
        #if it is predicting spam email as nonspam
        if(labels[i] == '1'):
            wrongPredictionCount += 1
            
print("Accuracy on test email")
print(100*(len(testEmail)-wrongPredictionCount)/len(testEmail))

def predictEmailSpamNotspam():
    file_list = glob.glob("test/*.txt")
    file_list.sort()
    emails = []
    #read all the text files of test folder
    for filename in file_list:
        with open(filename, 'r') as file:
            data = file.read().replace('\n', ' ')
            emails.append(data)
    wrongPredictionCount = 0
    special_characters=['~','`','!','@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']',':',';','"','|','\',','?','<','>',',','.','/','0','1','2','3','4','5','6','7','8','9',"'","Subject","subject"]
    for i in range(len(emails)):
        data = emails[i]
        #replace all the special character with empty string
        for sc in special_characters:
            data = data.replace(sc,'')
        words = data.split()
        #trim all the words and convert it to lowercase
        for j in range(len(words)):
            words[j] = words[j].strip().lower()
            
        isWordPresent = len(mostFrequentWords)*[0]
        for j in range(len(mostFrequentWords)):
            if mostFrequentWords[j] in words:
                isWordPresent[j] += 1
        #if svm predicts it as spam
        if(clf.predict([isWordPresent])[0] == '1'):
            print("file name", file_list[i])
            print("Predicted spam = +1")
        else:
            print("file name", file_list[i])
            print("Predicted not spam = 0")
            
            
predictEmailSpamNotspam()


# In[ ]:




