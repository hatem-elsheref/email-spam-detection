
# https://randerson112358.medium.com/email-spam-detection-using-python-machine-learning-abe38c889855
# https://blog.textedly.com/spam-text-message-examples

#Import libraries

import numpy as numpy
import pandas as panda
import nltk
from nltk.corpus import stopwords
import string

#Load the data


# dataFrame = panda.read_csv('db/emails.csv')
dataFrame = panda.read_csv('db/test.csv')
# print(dataFrame.head(5))


#Print the shape (Get the number of rows and cols)
result = dataFrame.shape
# print (result)
#Get the column names
dataFrame.columns

# print(dataFrame.columns)

#Checking for duplicates and removing them
dataFrame.drop_duplicates(inplace=True)
# result = dataFrame.shape
# print (result)

#Show the number of missing (NAN, NaN, na) data for each column
result = dataFrame.isnull().sum()
# print (result)


#Need to download stopwords
# nltk.download('stopwords')


# Tokenization (a list of tokens), will be used as the analyzer
# 1.Punctuations are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
# 2.Stop words in natural language processing, are useless words (data).
def process_text(text):
    # 1 Remove punctuation
    text_without_punctuation = [char for char in text if char not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)

    # 2 Remove Stop Words
    text_without_stop_words = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]

    # 3 Return a list of clean words
    return text_without_stop_words



#Show the Tokenization (a list of tokens )
# print (dataFrame['text'].head().apply(process_text))


# Convert the text into a matrix of token counts.

from sklearn.feature_extraction.text import CountVectorizer

messages_bow = CountVectorizer(analyzer=process_text).fit_transform(dataFrame['text'])

# print (messages_bow)


#Split data into 80% training & 20% testing data sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(messages_bow, dataFrame['spam'], test_size=0.20, random_state=0)



#Get the shape of messages_bow
# messages_bow.shape
# print (messages_bow.shape)


# Create and train the Multinomial Naive Bayes classifier which is suitable for classification with discrete features (e.g., word counts for text classification)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)

# #Print the predictions
# print(classifier.predict(X_train))
#
#
# print ('divider')
# # Print the actual values
# print(y_train.values)


#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train, pred))
print('Confusion Matrix: \n', confusion_matrix(y_train, pred))
print()
print('Accuracy: ', accuracy_score(y_train, pred))


# #Print the predictions
# print('Predicted value: ', classifier.predict(X_test))
# print ('divider')
# #Print Actual Label
# print('Actual value: ', y_test.values)


#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))









