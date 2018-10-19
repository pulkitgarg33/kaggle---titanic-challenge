import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



for dataset in [train , test]:
    dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    print(dataset['Title'].value_counts())
    print()
    
sns.catplot(x='Survived', y='Title', data=train, kind ='bar')

train.drop(['PassengerId' , 'Ticket'] , axis =1 , inplace = True)
test.drop(['Ticket'] , axis =1 , inplace = True)

test['Fare'].fillna(test[test['Pclass'] == 3].Fare.median(), inplace = True)


print(train[['Age','Title']].groupby('Title').mean())
sns.catplot(x='Age', y='Title', data=train, kind ='bar')

def getTitle(series):
    return series.str.split(',').str[1].str.split('.').str[0].str.strip()
print(getTitle(train[train.Age.isnull()].Name).value_counts())
mr_mask = train['Title'] == 'Mr'
miss_mask = train['Title'] == 'Miss'
mrs_mask = train['Title'] == 'Mrs'
master_mask = train['Title'] == 'Master'
dr_mask = train['Title'] == 'Dr'
train.loc[mr_mask, 'Age'] = train.loc[mr_mask, 'Age'].fillna(train[train.Title == 'Mr'].Age.mean())
train.loc[miss_mask, 'Age'] = train.loc[miss_mask, 'Age'].fillna(train[train.Title == 'Miss'].Age.mean())
train.loc[mrs_mask, 'Age'] = train.loc[mrs_mask, 'Age'].fillna(train[train.Title == 'Mrs'].Age.mean())
train.loc[master_mask, 'Age'] = train.loc[master_mask, 'Age'].fillna(train[train.Title == 'Master'].Age.mean())
train.loc[dr_mask, 'Age'] = train.loc[dr_mask, 'Age'].fillna(train[train.Title == 'Dr'].Age.mean())
print()
print(getTitle(train[train.Age.isnull()].Name).value_counts())

print(getTitle(test[test.Age.isnull()].Name).value_counts())
mr_mask = test['Title'] == 'Mr'
miss_mask = test['Title'] == 'Miss'
mrs_mask = test['Title'] == 'Mrs'
master_mask = test['Title'] == 'Master'
ms_mask = test['Title'] == 'Ms'
test.loc[mr_mask, 'Age'] = test.loc[mr_mask, 'Age'].fillna(test[test.Title == 'Mr'].Age.mean())
test.loc[miss_mask, 'Age'] = test.loc[miss_mask, 'Age'].fillna(test[test.Title == 'Miss'].Age.mean())
test.loc[mrs_mask, 'Age'] = test.loc[mrs_mask, 'Age'].fillna(test[test.Title == 'Mrs'].Age.mean())
test.loc[master_mask, 'Age'] = test.loc[master_mask, 'Age'].fillna(test[test.Title == 'Master'].Age.mean())
test.loc[ms_mask, 'Age'] = test.loc[ms_mask, 'Age'].fillna(test[test.Title == 'Miss'].Age.mean())
print(getTitle(test[test.Age.isnull()].Name).value_counts())




train['Title'] = train['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
test['Title'] = test['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
[df.drop(columns=['Name'], inplace = True) for df in [train, test]]
[train, test] = [pd.get_dummies(data = df, columns = ['Title']) for df in [train, test]]


for df in [train, test]:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] > 1).astype(int)


[df.drop(columns=['Cabin', 'SibSp', 'Parch'], inplace = True) for df in [train, test]]


for df in [train, test]:
    df.dropna(subset = ['Embarked'], inplace = True)
    
[train, test] = [pd.get_dummies(data = df, columns = ['Sex']) for df in [train, test]]
[train, test] = [pd.get_dummies(data = df, columns = ['Pclass']) for df in [train, test]]
[train, test] = [pd.get_dummies(data = df, columns = ['Embarked']) for df in [train, test]]

pass_id = test['PassengerId']
test.drop('PassengerId'  , axis =1 , inplace =True)
y_train = train['Survived']

train.drop('Survived' , axis =1 , inplace = True)

y_train = np.array(y_train)
y_train = y_train.reshape((889 , 1))


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(train , y_train)
y_pred = xgb.predict(test)


from sklearn.svm import SVC
svc = SVC()
svc.fit(train , y_train)
y_pred = svc.predict(test)

result = pd.DataFrame()
    
result['PassengerId'] = pass_id
result['Survived'] = y_pred

print(result['Survived'].value_counts())
result.to_csv('result.csv' , index = False)

