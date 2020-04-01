import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.ensemble import EasyEnsembleClassifier 
from sklearn.utils import resample
import joblib

df = pd.read_csv('app_train.csv')
df1 = pd.read_csv('app_test.csv')

dfNew = pd.DataFrame(df, columns = ['DAYS_AGE','ANNUITY','DAYS_WORK','INCOME','INCOME_TYPE'])
dfNew['TARGET'] = df['TARGET']
dfNew['TARGET_NAME'] = 'Late Payment'
dfNew['TARGET_NAME'][df['TARGET']==0] = 'Ontime Payment'
dfNew['ANNUITY'].fillna(df['ANNUITY'].median(), inplace=True)

dfNew1 = pd.DataFrame(df1, columns = ['DAYS_AGE','ANNUITY','DAYS_WORK','INCOME','INCOME_TYPE'])
dfNew1['TARGET'] = df1['TARGET']
dfNew1['TARGET_NAME'] = 'Late Payment'
dfNew1['TARGET_NAME'][df1['TARGET']==0] = 'Ontime Payment'

# print(dfNew.isnull().sum())
# print(dfNew1.isnull().sum())

dfDummy = pd.get_dummies(df['INCOME_TYPE'] ,prefix='INCOME_TYPE')
dfDummy1 = pd.get_dummies(df1['INCOME_TYPE'] ,prefix='INCOME_TYPE')

frames = [dfNew, dfDummy]
frames1 = [dfNew1, dfDummy1]

dfall = pd.concat(frames, axis=1)
dfall1 = pd.concat(frames1, axis=1)


# seperate the independent and target variable on training data
train_x = dfall.drop(columns=['TARGET','TARGET_NAME','INCOME_TYPE'],axis=1)
train_y = dfall['TARGET_NAME']
#print(train_x)
#print(train_y)

#print(train_x.columns.tolist())
# seperate the independent and target variable on testing data
test_x = dfall1.drop(columns=['TARGET','TARGET_NAME','INCOME_TYPE'],axis=1)
test_y = dfall1['TARGET_NAME']
# print(test_x)
# print(test_y)

# # separate minority and majority classes
# ontime = dfall[dfall['TARGET_NAME']=='Ontime Payment']
# late = dfall[dfall['TARGET_NAME']=='Late Payment']

# # upsample minority
# late_upsampled = resample(late,replace=True,n_samples=len(ontime))
                          
# # combine majority and upsampled minority
# upsampled = pd.concat([ontime, late_upsampled])

# train_x_upsampled = upsampled.drop(columns=['TARGET','TARGET_NAME','INCOME_TYPE'],axis=1)
# train_y_upsampled = upsampled['TARGET_NAME']
# test_x_upsampled = upsampled.drop(columns=['TARGET','TARGET_NAME','INCOME_TYPE'],axis=1)
# test_y_upsampled = upsampled['TARGET_NAME']

# model = LogisticRegression(solver='saga', penalty='l2', C=0.1)
model = EasyEnsembleClassifier()
model.fit(train_x, train_y)

joblib.dump(model, 'modelJoblib')