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
def predict(file):
    train=pd.read_csv(file,sep=';')



    pref_days = [0]
    for year in range(2020, 2021):
        pref_days.append(pd.Timestamp(f'{year}-12-31').dayofyear)

    for i in range(1, len(pref_days)):
        pref_days[i] += pref_days[i-1]


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


    train.subject_type.replace({"Автономный Округ": 'Республика',
                               "Автономная Область": 'Республика',
                               "Край": 'Область',
                               }, inplace=True)



    train['subject_or_city']='subject'
    train.loc[train['subject_type']=='Город','subject_or_city']='city'


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

    to_drop = []
    for i in train.columns:
        if (train[i].isna().sum() > len(train) * 0.15):
            to_drop.append(i)

    train = train.drop(columns=to_drop)

    for f in f_cols:
        if f in train.columns:
            train[f] = train[f].fillna(
                train.groupby('ind_date')[f].transform('median'))


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

    train.to_csv('train_games.csv',index=False)


    train=pd.get_dummies(train,columns=kitties)








    from catboost import CatBoostClassifier

    cb=CatBoostClassifier()#class_weights=[0.1,0.97])
    cb.load_model('model.h5')
    pred=cb.predict(train)
    return pred
