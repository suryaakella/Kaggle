import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df_train = pd.read_csv('/home/surya/Documents/kaggle_competitions/titanic/train.csv')
df_test = pd.read_csv('/home/surya/Documents/kaggle_competitions/titanic/test.csv')
print(df_train.shape)
print(df_test.shape)
print(df_train.isnull().sum())
print(df_test.isnull().sum())
sns.heatmap(df_train.isnull(),yticklabels = False,cbar = False)
# plt.show()
sns.heatmap(df_test.isnull(),yticklabels = False,cbar = False)
# plt.show()
df_train = df_train.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)
df_test = df_test.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
print(df_train.shape)
print(df_test.shape)
print(df_train.isnull().sum())
print(df_test.isnull().sum())
print(df_train.info)
print(pd.get_dummies(df_train))
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)
print(df_train.shape)
print(df_test.shape)
x_train = df_train.drop(['Survived'],axis=1)
y_train = df_train['Survived']
x_test = df_test
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0, stratify = y_train)
# import xgboost
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
pred = model.predict(df_test)
results = pd.read_csv('/home/surya/Documents/kaggle_competitions/titanic/gender_submission.csv')
results.iloc[:,1] = pred
print(results.head(10))
results.to_csv('/home/surya/Documents/kaggle_competitions/titanic/submission_by_surya.csv',index = False)
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)
pred = model.predict(df_test)
results = pd.read_csv('/home/surya/Documents/kaggle_competitions/titanic/gender_submission.csv')
results.iloc[:,1] = pred
print(results.head(10))
results.to_csv('/home/surya/Documents/kaggle_competitions/titanic/xg_by_surya.csv',index = False)

# accuracies = cross_val_score(model, x_train, y_train, cv = 10)
# print('vector of accuracies',accuracies)
# print('mean accuracy of k fold',accuracies.mean())
# print('function accuracy',accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))
