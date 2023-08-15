#!/usr/bin/env python
# coding: utf-8

# In[881]:


import pandas as pd
import numpy as np
import operator
import math
import random
import glob

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

#special character which is to be removed from the email
special_characters=['~','`','!','@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']',':',';','"','|','\',','?','<','>',',','.','/','0','1','2','3','4','5','6','7','8','9',"'","Subject","subject"]
for i in trainEmail:
    for sc in special_characters:
        #replace special character with empty string
        emails[i] = emails[i].replace(sc,'')
        
wordFrequency = {}
emailsAfterSplitting = len(emails)*[0]
for i in trainEmail:
    words = emails[i].split()
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
k=1000
#pick top 1000 words
mostFrequentWords = list(sortedWordFrequency.keys())[0:k]
#does email i contains word j
doesEmailContainWord = np.zeros(shape=(len(emails),len(mostFrequentWords)))

for i in trainEmail:
    for j in range(len(mostFrequentWords)):
        if mostFrequentWords[j] in emailsAfterSplitting[i]:
            doesEmailContainWord[i][j] = 1
            
countNonspam = 0
countSpam = 0
for i in trainEmail:
    if(labels[i] == '0'):
        countNonspam += 1
    else:
        countSpam += 1
p = countSpam/len(trainEmail)

#calculate probability of words belonging to non-spam mails
nonSpamWordProbability = k*[0]
#calculate probability of words belonging to spam mails
spamWordProbability = k*[0]
for j in range(k):
    noNonSpamEmailsContainsWord = 0
    noSpamEmailsContainsWord = 0
    for i in trainEmail:
        if(labels[i] == '0' and doesEmailContainWord[i][j] == 1):
            noNonSpamEmailsContainsWord = noNonSpamEmailsContainsWord+1
        elif(labels[i] == '1' and doesEmailContainWord[i][j] == 1):
            noSpamEmailsContainsWord = noSpamEmailsContainsWord+1
    nonSpamWordProbability[j] = noNonSpamEmailsContainsWord/countNonspam
    spamWordProbability[j] = noSpamEmailsContainsWord/countSpam
    
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
    spamProbability = 0
    nonSpamProbability = 0
    for word in words:
        if word in mostFrequentWords:
            if(spamWordProbability[mostFrequentWords.index(word)] != 0):
                spamProbability += math.log(spamWordProbability[mostFrequentWords.index(word)])
            if(nonSpamWordProbability[mostFrequentWords.index(word)] != 0):
                nonSpamProbability += math.log(nonSpamWordProbability[mostFrequentWords.index(word)])

    spamProbability += math.log(p)
    nonSpamProbability += math.log(1-p)
    
    if(spamProbability > nonSpamProbability):
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
    #read all the files
    for filename in file_list:
        with open(filename, 'r') as file:
            data = file.read().replace('\n', ' ')
            emails.append(data)
    wrongPredictionCount = 0
    special_characters=['~','`','!','@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']',':',';','"','|','\',','?','<','>',',','.','/','0','1','2','3','4','5','6','7','8','9',"'","Subject","subject"]
    spamWords = ['click','open','http','hurry','money','free','discount','earn','cash','price']
    for i in range(len(emails)):
        data = emails[i]
        #replace special character with empty string
        for sc in special_characters:
            data = data.replace(sc,'')
        words = data.split()
        for j in range(len(words)):
            words[j] = words[j].strip().lower()
        spamProbability = 0
        nonSpamProbability = 0
        for word in words:
            #give more preference to the words which are in spamWords list
            if word in spamWords:
                spamProbability += math.log(0.95)
                nonSpamProbability += math.log(0.01)
            elif word in mostFrequentWords:
                if(spamWordProbability[mostFrequentWords.index(word)] != 0):
                    spamProbability += math.log(spamWordProbability[mostFrequentWords.index(word)])
                if(nonSpamWordProbability[mostFrequentWords.index(word)] != 0):
                    nonSpamProbability += math.log(nonSpamWordProbability[mostFrequentWords.index(word)])

        spamProbability += math.log(p)
        nonSpamProbability += math.log(1-p)

        if(spamProbability > nonSpamProbability):
            print("file name", file_list[i])
            print("Predicted spam = +1")
        else:
            print("file name", file_list[i])
            print("Predicted not spam = 0")
            
            
predictEmailSpamNotspam()

