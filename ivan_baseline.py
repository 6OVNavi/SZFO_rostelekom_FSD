# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random,os
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
seed=42

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed)

train=pd.read_csv('train.csv',sep=';')
val=pd.read_csv('test.csv',sep=';')



print()
euro=pd.read_excel('euro.xlsx')
dollar=pd.read_excel('dollar.xlsx')
print(euro)
euro['data']=  euro['data'].apply(lambda x: str(x)[:-9].split('-')[0]+'-'+str(x)[:-9].split('-')[1]+'-'+str(x)[:-9].split('-')[2])
dollar['data']=  dollar['data'].apply(lambda x: str(x)[:-9].split('-')[0]+'-'+str(x)[:-9].split('-')[1]+'-'+str(x)[:-9].split('-')[2])

euro.columns = euro.columns.str.replace('data', 'period')
dollar.columns = dollar.columns.str.replace('data', 'period')

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

add_columns = ['year', 'month',  'dayofyear', 'season', 'ind_date']
euro[add_columns] = list(euro['period'].apply(prepare))
for col in add_columns:
    euro[col] = euro[col].astype(np.int32)
euro = euro.sort_values(by='ind_date')
euro=euro.set_index('ind_date')

add_columns = ['year', 'month',  'dayofyear', 'season', 'ind_date']
dollar[add_columns] = list(dollar['period'].apply(prepare))
for col in add_columns:
    dollar[col] = dollar[col].astype(np.int32)
dollar = dollar.sort_values(by='ind_date')
dollar=dollar.set_index('ind_date')

#print(euro)

'''def get_curs_euro_now(ind_date):
    return np.nanmean(euro.loc[ind_date-7:ind_date]['curs'])
def get_curs_euro_before(ind_date):
    return np.nanmean(euro.loc[ind_date-14:ind_date-8]['curs'])
def get_curs_dollar_now(ind_date):
    return np.nanmean(dollar.loc[ind_date-7:ind_date]['curs'])
def get_curs_dollar_before(ind_date):
    return np.nanmean(dollar.loc[ind_date-14:ind_date-8]['curs'])


train['euro_curs_cur']=train['ind_date'].apply(lambda x: get_curs_euro_now(x))
train['euro_curs_last_week']=train['ind_date'].apply(lambda x: get_curs_euro_before(x))
train['dollar_curs_cur']=train['ind_date'].apply(lambda x: get_curs_dollar_now(x))
train['dollar_curs_last_week']=train['ind_date'].apply(lambda x: get_curs_dollar_before(x))

val['euro_curs_cur']=val['ind_date'].apply(lambda x: get_curs_euro_now(x))
val['euro_curs_last_week']=val['ind_date'].apply(lambda x: get_curs_euro_before(x))
val['dollar_curs_cur']=val['ind_date'].apply(lambda x: get_curs_dollar_now(x))
val['dollar_curs_last_week']=val['ind_date'].apply(lambda x: get_curs_dollar_before(x))

print(train.isna().sum())
print(val.isna().sum())'''

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

kitties=['subject_type','subject_name','city_name','subject_or_city','hex']
#print(len(train))
#print(train)
to_drop = []
for i in range(len(train.columns)):
    a = train[train.columns[i]].value_counts().values[0]
    a = a / len(train) * 100
    # print(train.columns[i])
    # print(a)
    if a > 75 and train.columns[i] not in ['subject_or_city','label','euro_curs_cur','euro_curs_last_week',
                                           'dollar_curs_cur','dollar_curs_last_week',*kitties]:
        to_drop.append(train.columns[i])

train = train.drop(columns=to_drop)
val = val.drop(columns=to_drop)

to_drop = []
for i in train.columns:
    if (train[i].isna().sum() > len(train) * 0.15):
        to_drop.append(i)

train = train.drop(columns=to_drop)
val = val.drop(columns=to_drop)

for f in f_cols:
    if f in train.columns:
        train[f] = train[f].fillna(
            train.groupby('ind_date')[f].transform('median'))
        val[f] = val[f].fillna(
            val.groupby('ind_date')[f].transform('median'))

train['rate']=0.0
for i in range(len(train.subject_name.value_counts().keys())):
    train.loc[train['subject_name']==train.subject_name.value_counts().keys()[i], 'rate']=train[train['subject_name']==train.subject_name.value_counts().keys()[0]]['label'].sum()/train.subject_name.value_counts()[i]


print(train.columns)
'''sns.set_theme(font_scale=0.6)
plt.figure(figsize=(10, 10))
sns.heatmap(train.corr(), annot=True,linewidths=1)
plt.tight_layout()
plt.show()
'''

games=['alchemist','lostark','tanks','warface','warplane','warship']

def twitch_viewers_now(ind_date):
    return np.nanmean(df2.loc[ind_date-7:ind_date]['Twitch Viewers'])
def twitch_viewers_before(ind_date):
    return np.nanmean(df2.loc[ind_date-14:ind_date-8]['Twitch Viewers'])
def players_now(ind_date):
    return np.nanmean(df2.loc[ind_date-7:ind_date]['Players'])
def players_before(ind_date):
    return np.nanmean(df2.loc[ind_date-14:ind_date-8]['Players'])


for game in games:
    df2=pd.read_csv(f'{game}.csv', sep=';')
    df2.DateTime=df2.DateTime.apply(lambda x: x.split(' ')[0])
    df2=df2.rename(columns={'DateTime': 'period'})

    for i in range(len(df2)):
        if int(df2.period[i].split('-')[0]) < 2021:
            df2 = df2.drop(i, axis=0)
    df2 = df2.reset_index(drop=True)

    df2[add_columns] = list(df2['period'].apply(prepare))

    for col in add_columns:
        df2[col] = df2[col].astype(np.int32)
    df2 = df2.sort_values(by='ind_date')
    df2 = df2.set_index('ind_date')

    train[f'twitch_viewers_{game}_now'] = train['ind_date'].apply(lambda x: twitch_viewers_now(x))
    train[f'twitch_viewers_{game}_last_week'] = train['ind_date'].apply(lambda x: twitch_viewers_before(x))
    train[f'players_{game}_now'] = train['ind_date'].apply(lambda x: players_now(x))
    train[f'players_{game}_last_week'] = train['ind_date'].apply(lambda x: players_before(x))

    val[f'twitch_viewers_{game}_now'] = val['ind_date'].apply(lambda x: twitch_viewers_now(x))
    val[f'twitch_viewers_{game}_last_week'] = val['ind_date'].apply(lambda x: twitch_viewers_before(x))
    val[f'players_{game}_now'] = val['ind_date'].apply(lambda x: players_now(x))
    val[f'players_{game}_last_week'] = val['ind_date'].apply(lambda x: players_before(x))

train.to_csv('train_games.csv',index=False)
val.to_csv('val_games.csv',index=False)


train=pd.get_dummies(train,columns=kitties)
val=pd.get_dummies(val,columns=kitties)




for i in val.columns:
    if i not in train.columns:
        train[i]=0
for i in train.columns:
    if i not in val.columns:
        val[i]=0

train=train.sort_index(axis=1)
val=val.sort_index(axis=1)



X_train=train.drop('label',axis=1)
y_train=train['label']

X_val=val.drop('label',axis=1)
y_val=val['label']

from catboost import CatBoostClassifier

cb=CatBoostClassifier(iterations=10000,eval_metric='Precision:use_weights=False',random_state=seed,auto_class_weights='Balanced')#class_weights=[0.1,0.97])
cb.fit(X_train,y_train,eval_set=(X_val,y_val))
print(cb.best_score_)
pred=cb.predict(X_val)
from sklearn.metrics import precision_score, accuracy_score,recall_score
#print(precision_score(pred,y_val['label']),'-------precision')
print(precision_score(y_val,pred),'-------precision')
print(accuracy_score(y_val,pred),'-------accuracy')
print(recall_score(y_val,pred),'-------recall')
