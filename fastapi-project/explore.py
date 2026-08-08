# Generated from: SPAM_predictor.ipynb
# Converted at: 2026-08-07T08:24:54.098Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("emails.csv (1).zip")

data.head(200)

data['spam'].value_counts()

data.isnull().sum()

data.columns

data.shape

X=data['text']
y=data['spam']

X

y

type(X)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(X)

X

type(X)

X.shape

X=X.toarray()

featurenames=cv.get_feature_names_out()

featurenames[:3000]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

from sklearn.naive_bayes import MultinomialNB
gnb=MultinomialNB()
gnb.fit(X_train,y_train)

pred=gnb.predict(X_test)

pred

prob=gnb.predict_proba(X_test)
prob

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

email=input("enter the email that you got:")
email_vec=cv.transform([email])
prediction=gnb.predict(email_vec)
if prediction==1:
    print("Alert!it is a spam")
else:
    print("it is not spam")