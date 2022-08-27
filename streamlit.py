import streamlit as st
import pandas as pd
import numpy as np
from base import predict
from map import region, label
import matplotlib
import matplotlib.pyplot as plt
import os


st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(suppress_st_warning=True)
def upload_file(file):
    pred=predict(file)
    return  pred

uploaded_file = st.file_uploader(label='Выберите датасет:')
pred = upload_file(uploaded_file)

pred=pd.DataFrame(pred, columns=['label'])
table=pd.read_csv(uploaded_file, sep=';')
if 'label' in table.columns:
   table=table.drop('label', axis=1)
table=pd.concat([table, pred], axis=1)

train=table.copy()
train['subject_type']=train['subject_type'].apply(lambda x: x.lower() if x!='Республика' else x)
train.loc[train['subject_type']=='область', 'subject']=train['subject_name']+' '+train['subject_type']
train.loc[train['subject_type']=='край', 'subject']=train['subject_name']+' '+train['subject_type']
train.loc[train['subject_type']=='Республика', 'subject']=train['subject_name']
train.loc[train['subject_type']=='город', 'subject']=train['subject_name']
train.loc[train['subject_type']=='автономный округ', 'subject']=train['subject_name']+' '+train['subject_type']
train.loc[train['subject_type']=='автономная область', 'subject']=train['subject_name']+' '+train['subject_type']

gr=train[train['label']==1]

graph=pd.DataFrame({'label_amount':gr.subject.value_counts()[:5], 'subject': gr.subject.value_counts().keys()[:5]}).reset_index(drop=True)

def main():

    result = st.button('Предсказать')
    if result:
        st.write(table)
    gra=st.sidebar.button('Показать график')
    if gra:
        graph.plot(x='subject', y='label_amount', kind='bar', figsize=(12, 10), xlabel='subject', ylabel='label_amount')
        st.pyplot()
    mapa=st.sidebar.selectbox('Выбери карту', ('С метками',  'С регионами'))
    if mapa=='С регионами':
        to_map = pd.DataFrame({'amount_connect': list(gr.subject.value_counts()),
                               'subject': list(gr.subject.value_counts().keys())})
        to_map = to_map.sort_values(by='subject')
        map=region(to_map)
        st.write(map)
    if mapa=='С метками':
        map=label(gr)
        

main()
