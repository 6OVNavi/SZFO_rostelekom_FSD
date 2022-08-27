import streamlit as st
import pandas as pd
import numpy as np
from base import predict
import matplotlib
import matplotlib.pyplot as plt
import os


st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(suppress_st_warning=True)
def upload_file(file):
    pred=predict(file)
    return  pred
uploaded_file = st.file_uploader(label='Выберите датасет:')
pred=upload_file(uploaded_file)
pred=pd.DataFrame(pred, columns=['label'])
table=pd.read_csv(uploaded_file, sep=';')
if 'label' in table.columns:
   table=table.drop('label', axis=1)
table=pd.concat([table, pred], axis=1)

train=table.copy()
train.loc[train['subject_type']=='Область', 'subject']=train['subject_name']+' '+train['subject_type']
train.loc[train['subject_type']=='Край', 'subject']=train['subject_name']+' '+train['subject_type']
train.loc[train['subject_type']=='Республика', 'subject']=train['subject_type']+' '+train['subject_name']
train.loc[train['subject_type']=='Город', 'subject']=train['subject_name']
train.loc[train['subject_type']=='Автономный Округ', 'subject']=train['subject_name']+' '+train['subject_type']
train.loc[train['subject_type']=='Автономная Область', 'subject']=train['subject_name']+' '+train['subject_type']

gr=train[train['label']==1]
graph=pd.DataFrame({'label_amount':gr.subject.value_counts()[:5], 'subject': gr.subject.value_counts().keys()[:5]}).reset_index(drop=True)

def main():

    result = st.button('Предсказать')
    if result:
        st.write(table)
        graph.plot(x='subject', y='label_amount', kind='bar', figsize=(12, 10), xlabel='subject', ylabel='label_amount')
        st.pyplot()
    

main()
