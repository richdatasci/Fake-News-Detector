import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Read the data
df = pd.read_csv(r'E:\Users\Richie\Desktop\ML Projects\FakeNewsDetector\news.csv') #ensure the full path to the data is inserted here if it is stored locally. 
#Get shape and head
df.shape
df.head()
#get the labels
labels = df.label
labels.head()

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
# this code fits and transforms train set and transforms the test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%') 

#Build confusion matrix
matrix = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(matrix)

# output a classification report which can be exported as a .csv
report = classification_report(y_test, y_pred, output_dict=True)
reportdata = pd.DataFrame(report).transpose()
print(reportdata)