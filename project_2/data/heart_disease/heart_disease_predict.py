import os
import pandas as pd
import numpy as np

os.chdir('C:\Users\WZU448\ct16_cap1_ds4\project_2\data\heart_disease')

col_names=['age','sex','cp','trestbps','chol','fbs','restecg',
'thalach','exang','oldpeak','slope','ca','thal','num']

df_cle = pd.read_csv('processed.cleveland.data', header=None, names=col_names, na_values='?')
df_hung = pd.read_csv('processed.hungarian.data', header=None, names=col_names, na_values='?')
df_switz = pd.read_csv('processed.switzerland.data', header=None, names=col_names, na_values='?')
df_va = pd.read_csv('processed.va.data', header=None, names=col_names, na_values='?')

df = pd.concat([df_cle, df_hung, df_switz, df_va])

#Dummy - cp, restecg, slope, ca, thal
#Interactions - oldpeak*slope