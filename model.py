
import pandas as panda
from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# nltk.download('stopwords')


ROOT_PATH = 'db/'
MAIN_FILE = 'emails.csv'
# MAIN_FILE = 'test.csv'
FULL_PATH = ROOT_PATH + MAIN_FILE
TEST_SIZE = 20


def loadFromDataSet(filename):
    dataframe = panda.read_csv(filename)
    return dataframe

def removeDuplicates(dataframe):
    dataframe.drop_duplicates(inplace=True)
    return dataframe

def filterTheText(text):

    # 1 Remove punctuation
    text_without_punctuation = [char for char in text if char not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)

    # 2 Remove Stop Words
    text_without_stop_words = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]

    # 3 Return a list of clean words
    return text_without_stop_words

def getVectorizer():
    return CountVectorizer(analyzer=filterTheText)

def tokenizerTheString(dataframe, vectorize):
    return vectorize.fit_transform(dataframe['text'])

def getTrainAndTest(features, labels , size):
    return train_test_split(features, labels, test_size=size, random_state=0)

def trainTheModel(X_TRAIN, Y_TRAIN):
    classifierModel = MultinomialNB()
    classifierModel.fit(X_TRAIN, Y_TRAIN)
    return classifierModel

def report(labels, modelPrediction):
    print(classification_report(labels, modelPrediction))
    print('Confusion Matrix: \n', confusion_matrix(labels, modelPrediction))
    print()
    print('Accuracy: ', accuracy_score(labels, modelPrediction))


VECTORIZER = getVectorizer()

def run():
    dataFrame = loadFromDataSet(FULL_PATH)
    dataFrame = removeDuplicates(dataFrame)
    messages = tokenizerTheString(dataFrame, VECTORIZER)
    training_features, testing_features, trainingLabels, testingLabels = getTrainAndTest(messages, dataFrame['spam'], TEST_SIZE)
    classifier = trainTheModel(training_features, trainingLabels)
    return classifier


# prediction = classifier.predict(training_features)
# report(trainingLabels, prediction)
# prediction = classifier.predict(testing_features)
# report(testingLabels, prediction)


def check(classifierModel, vectorize, message):

    readyData = vectorize.transform([message]).toarray()
    return classifierModel.predict(readyData)





SPAM = "amazon is sending you a refund of $32.64. Please reply with your bank account and routing number to receive your refund"
HAM = "hi ali"



res = check(run(), VECTORIZER, HAM)[0]
print (res)
