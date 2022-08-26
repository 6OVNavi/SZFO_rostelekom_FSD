
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random,os

seed=42

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed)

train=pd.read_csv('train_all.csv')
ID=np.linspace(1,len(train),len(train)).astype(int)
train['ID']=ID

print(list(train.columns))
#число дней до текущего года
pref_days = [0]
for year in range(2020, 2021):
    pref_days.append(pd.Timestamp(f'{year}-12-31').dayofyear)
print(pref_days)
for i in range(1, len(pref_days)):
    pref_days[i] += pref_days[i-1]
print(pref_days)

#признаки из datetime
def prepare(x):
    date = x.split('-')
    #time = x[1].split(':')
    year,month,day = map(int, date)
    timestamp = pd.Timestamp(f'{year}-{month}-{day}')
    #hour = int(time[0])
    dayofyear = timestamp.dayofyear
    dayofweek = timestamp.dayofweek
    ind_date = dayofyear + pref_days[year - 2021] - 1
    #ind_hour = ind_date * 24 + hour
    if dayofweek >= 5:
        days_to_weekend = 0
    else:
        days_to_weekend = min(dayofweek + 1, abs(5 - dayofweek))
    return year, month,  dayofyear,  (month-1) // 3, ind_date

add_columns = ['year', 'month',  'dayofyear', 'season', 'ind_date']
train[add_columns] = list(train['period'].apply(prepare))
for col in add_columns:
    train[col] = train[col].astype(np.int32)
train = train.sort_values(by='ind_date')
train=train.drop('period',axis=1)

train.subject_type.replace({"Автономный Округ": 'Республика',
                           "Автономная Область": 'Республика',
                           "Край": 'Область',
                           }, inplace=True)



train['subject_or_city']='subject'
train.loc[train['subject_type']=='Город','subject_or_city']='city'







train=train.dropna()



kitties=['subject_type','subject_name','city_name','subject_or_city']
train=pd.get_dummies(train,columns=kitties)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
le.fit(train['hex'])
train['hex']=le.transform(train['hex'])

from sklearn.model_selection import train_test_split



X=train.drop('label',axis=1)
y=train[['label','ID']]
#print(train['hex'])



#print(list(X.columns))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

X_temp=X_train.merge(y_train,on='ID')

X1=X_temp[X_temp['label']==1]
X2=X_temp[X_temp['label']==0]
fraction=len(X2)//len(X1)
val_tests=np.zeros(len(X_val))
#pred=np.zeros(len(test))
print(len(X1),len(X2),'-@@-')

X_val=X_val.drop('ID',axis=1)

from catboost import CatBoostClassifier

#model = CatBoostClassifier(iterations=300, use_best_model=True, eval_metric='Precision', random_state=seed,learning_rate=0.001)

best_its=[]

for z in range(fraction):
    X2_2=X2[z*len(X1):(z+1)*len(X1)]
    X_fin=pd.concat([X1,X2_2])
    print(z*len(X1),(z+1)*len(X1))
    X_tr=X_fin.drop(columns=y_train.columns)
    y_tr=X_fin.drop(columns=X_train.columns)

    model = CatBoostClassifier(iterations=300, use_best_model=True, eval_metric='Precision', random_state=seed,learning_rate=0.001,)

    model.fit(X_tr,y_tr['label'],eval_set=(X_val,y_val['label']),verbose=0)
    val_test=np.array(model.predict(X_val))
    val_tests=val_test+val_tests

    #test_pr=model.predict(test)
    #pred=pred+test_pr
    best_its.append(
        (model.best_score_['learn']['Precision'], model.best_score_['validation']['Precision'],
         model.best_iteration_))
val_tests=val_tests/fraction
val_tests=list(map(lambda x: round(x),val_tests))
print()

from sklearn.metrics import precision_score

print(precision_score(y_val['label'],val_tests),'-------precision')
#pred=pred/fraction
#pred = list(map(lambda x: round(x), pred))



