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

train=pd.read_csv('train.csv',sep=';')
val=pd.read_csv('test.csv',sep=';')

merged=pd.read_csv('train_all.csv')

print(list(train.columns))

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

add_columns = ['year', 'month',  'dayofyear', 'season', 'ind_date']
val[add_columns] = list(val['period'].apply(prepare))
for col in add_columns:
    val[col] = val[col].astype(np.int32)
val = val.sort_values(by='ind_date')
val=val.drop('period',axis=1)

train.subject_type.replace({"Автономный Округ": 'Республика',
                           "Автономная Область": 'Республика',
                           "Край": 'Область',
                           }, inplace=True)
val.subject_type.replace({"Автономный Округ": 'Республика',
                           "Автономная Область": 'Республика',
                           "Край": 'Область',
                           }, inplace=True)



train['subject_or_city']='subject'
train.loc[train['subject_type']=='Город','subject_or_city']='city'

val['subject_or_city']='subject'
val.loc[val['subject_type']=='Город','subject_or_city']='city'

f_cols=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30']

maybe_imbalance=['f30','f23','f14','f5']

#train=train.drop(columns=maybe_imbalance)
to_drop = []
for i in range(len(train.columns)):
    a = train[train.columns[i]].value_counts().values[0]
    a = a / len(train) * 100
    # print(train.columns[i])
    # print(a)
    if a > 75 and train.columns[i] not in ['subject_or_city','label']:
        to_drop.append(train.columns[i])

train = train.drop(columns=to_drop)
val = val.drop(columns=to_drop)
print(list(train.columns))

for f in f_cols:
    if f in train.columns:
        train[f] = train[f].fillna(
            train.groupby('ind_date')[f].transform('median'))
        val[f] = val[f].fillna(
            val.groupby('ind_date')[f].transform('median'))

kitties=['subject_type','subject_name','city_name','subject_or_city','hex']
train=pd.get_dummies(train,columns=kitties)
val=pd.get_dummies(val,columns=kitties)

from sklearn.preprocessing import LabelEncoder

'''le=LabelEncoder()
le.fit(train['hex'])
train['hex']=le.transform(train['hex'])'''


X_train=train.drop('label',axis=1)
y_train=train['label']

X_val=train.drop('label',axis=1)
y_val=train['label']

from catboost import CatBoostClassifier

cb=CatBoostClassifier(eval_metric='Precision',random_state=seed,auto_class_weights='Balanced')#class_weights=[0.1,0.97])
cb.fit(X_train,y_train,eval_set=(X_val,y_val))
print(cb.best_score_)
pred=cb.predict(X_val)
from sklearn.metrics import precision_score, accuracy_score,recall_score
#print(precision_score(pred,y_val['label']),'-------precision')
print(precision_score(y_val['label'],pred),'-------precision')
print(accuracy_score(y_val['label'],pred),'-------accuracy')
print(recall_score(y_val['label'],pred),'-------recall')
'''X_temp=X_train.merge(y_train,on='ID')

X1=X_temp[X_temp['label']==1]
X2=X_temp[X_temp['label']==0]
fraction=len(X2)//len(X1)
val_tests=np.zeros(len(X_val))
#pred=np.zeros(len(test))
print(len(X1),len(X2),'-@@-')

X_val=X_val.drop('ID',axis=1)



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



print(precision_score(y_val['label'],val_tests),'-------precision')
print(best_its)
#pred=pred/fraction
#pred = list(map(lambda x: round(x), pred))
'''


