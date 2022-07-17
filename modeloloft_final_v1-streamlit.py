#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import streamlit as st
import string
from fuzzywuzzy import fuzz


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df_aptos = pd.read_csv('caracteristicas_aptos.csv')
df_fotos = pd.read_csv('fotos.csv')


# In[4]:


df_aptos.head()


# In[5]:


df_aptos.columns


# In[6]:


df_aptos = df_aptos[['id', 'area', 'price', 'bedrooms', 'parking_spots', 'living_room_view_is_blocked', 'floor',
       'built_year', 'valor_condominio', 'subway_shortest_distance',
       'created_at', 'status', 'is_published','last_unpublished_at', 'unpublish_reason', 'sold_at',
       'polygon_name', 'building_id', 'unit_id', 'city',
       'latitude', 'longitude']]


# In[7]:


df_aptos.rename(columns={'price':'preco','bedrooms':'quartos','parking_spots':'vagasGaragem',
                         'living_room_view_is_blocked':'vistaSalaBloqueada', 'floor':'andar','built_year':'anoConstrucao',
                         'subway_shortest_distance':'menorDistanciaMetro','created_at':'criadoEm','is_published':'publicado',
                         'last_unpublished_at':'despublicadoEm','unpublish_reason':'motivoDespublicacao','sold_at':'vendidoEm',
                         'polygon_name':'regiao','building_id':'idPredio','unit_id':'idApartamento','city':'cidade'
                        }, inplace=True)
df_aptos.head()


# In[8]:


df_fotos.head()


# In[9]:


df_fotos.columns


# In[10]:


df_fotos = df_fotos.drop('max(ref_date)',axis=1)


# In[11]:


df_fotos.rename(columns={'listing_id':'id','there_is_bathroom':'temBanheiro','there_is_kitchen':'temCozinha'}, inplace=True)
df_fotos.head()


# In[12]:


df_aptos.shape


# In[13]:


df_aptos.isna().sum()


# In[14]:


df_aptos = df_aptos[(df_aptos.vistaSalaBloqueada.notna()) & (df_aptos.anoConstrucao.notna()) & 
                    (df_aptos.valor_condominio.notna()) & (df_aptos.menorDistanciaMetro.notna()) &
                    (df_aptos.regiao.notna()) & (df_aptos.idPredio.notna()) & (df_aptos.idApartamento.notna()) &
                   (df_aptos.cidade.notna()) & (df_aptos.latitude.notna()) & (df_aptos.longitude.notna())]


# In[15]:


df_aptos.isna().sum()


# In[16]:


df_aptos.shape


# In[17]:


df_fotos.shape


# In[18]:


df_fotos.isna().sum()


# In[19]:


df_fotos.dropna(inplace=True)


# In[20]:


df_fotos.shape


# In[21]:


df_fotos.isna().sum()


# In[22]:


df_aptos.duplicated().sum()


# In[23]:


df_fotos.duplicated().sum()


# In[24]:


df_aptos.duplicated(subset=['id']).sum()


# In[25]:


df_fotos.duplicated(subset=['id']).sum()


# In[26]:


df_fotos.drop_duplicates(subset=['id'],inplace=True)


# In[27]:


df_fotos.shape


# In[28]:


df = df_aptos.merge(df_fotos, how='inner')


# In[29]:


df.shape


# In[30]:


df = df[df['cidade'] == 'São Paulo']


# In[31]:


df = df.drop('cidade',axis=1)


# In[32]:


df.shape


# In[33]:


df.describe()


# In[34]:


df['area'].nlargest(10)


# In[35]:


df = df[df['area']<1500]


# In[36]:


round(df['preco'].nlargest(100))


# In[37]:


df['preco'].hist(bins=[0, 250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000, 2250000, 2500000, 2750000, 3000000])


# In[38]:


df['preco'].nsmallest()


# In[39]:


df = df[df['preco']<3000000]


# In[40]:


df.shape


# In[41]:


df['andar'].hist(bins=[0,5,10,15,20,30,40])


# In[42]:


df = df[df['andar']<30]


# In[43]:


df.shape


# In[44]:


df = df[(df['anoConstrucao']<=2022) | (df['anoConstrucao']>1900)]


# In[45]:


df.shape


# In[46]:


df = df[df['valor_condominio'] <= 10000]


# In[47]:


df = df[df['menorDistanciaMetro'] <= 20000]


# In[48]:


df = df[df['vagasGaragem'] < 6]


# In[49]:


df['anoConstrucao'] = df['anoConstrucao'].astype('int64')


# In[50]:


df.shape


# In[51]:


df['status'].unique()


# In[52]:


df['motivoDespublicacao'].unique()


# In[53]:


df['motivoDespublicacao'].value_counts()


# In[54]:


df[['criadoEm','despublicadoEm','vendidoEm']] = df[['criadoEm','despublicadoEm','vendidoEm']].apply(pd.to_datetime)


# In[55]:


prazoVenda1 = 30
prazoVenda2 = 60
prazoVenda3 = 90
prazoVenda4 = 120

df['prazoVenda1'] = df['criadoEm'] + pd.DateOffset(days=prazoVenda1)
df['prazoVenda2'] = df['criadoEm'] + pd.DateOffset(days=prazoVenda2)
df['prazoVenda3'] = df['criadoEm'] + pd.DateOffset(days=prazoVenda3)
df['prazoVenda4'] = df['criadoEm'] + pd.DateOffset(days=prazoVenda4)


# In[56]:


df[['criadoEm','prazoVenda1']].head()


# In[57]:


max(df['criadoEm']),max(df['despublicadoEm'][df['despublicadoEm'].notnull()]),max(df['vendidoEm'][df['vendidoEm'].notnull()])


# In[58]:


dataReferencia = max(df['criadoEm'])
dataCorte1 = pd.to_datetime(dataReferencia) - pd.DateOffset(days=prazoVenda1)
dataCorte2 = pd.to_datetime(dataReferencia) - pd.DateOffset(days=prazoVenda2)
dataCorte3 = pd.to_datetime(dataReferencia) - pd.DateOffset(days=prazoVenda3)
dataCorte4 = pd.to_datetime(dataReferencia) - pd.DateOffset(days=prazoVenda4)


# In[59]:


df1 = df[df['criadoEm'] < dataCorte1]
df2 = df[df['criadoEm'] < dataCorte2]
df3 = df[df['criadoEm'] < dataCorte3]
df4 = df[df['criadoEm'] < dataCorte4]


# In[60]:


df1['situacao'] = df.apply(lambda x: 'vendido' if (x['status'] == 'SOLD' or x['motivoDespublicacao'] == 'UNIT_SOLD') and x['vendidoEm'] <= x['prazoVenda1'] else 'excluir', axis=1)
df2['situacao'] = df.apply(lambda x: 'vendido' if (x['status'] == 'SOLD' or x['motivoDespublicacao'] == 'UNIT_SOLD') and x['vendidoEm'] <= x['prazoVenda2'] else 'excluir', axis=1)
df3['situacao'] = df.apply(lambda x: 'vendido' if (x['status'] == 'SOLD' or x['motivoDespublicacao'] == 'UNIT_SOLD') and x['vendidoEm'] <= x['prazoVenda3'] else 'excluir', axis=1)
df4['situacao'] = df.apply(lambda x: 'vendido' if (x['status'] == 'SOLD' or x['motivoDespublicacao'] == 'UNIT_SOLD') and x['vendidoEm'] <= x['prazoVenda4'] else 'excluir', axis=1)


# In[61]:


df1['situacao'].value_counts(), df2['situacao'].value_counts(),df3['situacao'].value_counts(),df4['situacao'].value_counts()


# In[62]:


df1 = df1[df1.situacao != 'excluir']
df2 = df2[df2.situacao != 'excluir']
df3 = df3[df3.situacao != 'excluir']
df4 = df4[df4.situacao != 'excluir']


# In[63]:


df1['situacao'].value_counts(), df2['situacao'].value_counts(),df3['situacao'].value_counts(),df4['situacao'].value_counts()


# In[64]:


df.head()


# In[65]:


df.columns


# In[79]:


yPreco1 = df1['preco'].reset_index(drop=True)
XPreco1 = df1[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude','anoConstrucao', 'regiao', 'qualidade', 'angulo', 'bagunca', 
                               'conservacao', 'temBanheiro', 'temCozinha','vistaSalaBloqueada']].reset_index(drop=True)
yPreco2 = df2['preco'].reset_index(drop=True)
XPreco2 = df2[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude','anoConstrucao', 'regiao', 'qualidade', 'angulo', 'bagunca', 
                               'conservacao', 'temBanheiro', 'temCozinha','vistaSalaBloqueada']].reset_index(drop=True)
yPreco3 = df3['preco'].reset_index(drop=True)
XPreco3 = df3[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude','anoConstrucao', 'regiao', 'qualidade', 'angulo', 'bagunca', 
                               'conservacao', 'temBanheiro', 'temCozinha','vistaSalaBloqueada']].reset_index(drop=True)
yPreco4 = df4['preco'].reset_index(drop=True)
XPreco4 = df4[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude','anoConstrucao', 'regiao', 'qualidade', 'angulo', 'bagunca', 
                               'conservacao', 'temBanheiro', 'temCozinha','vistaSalaBloqueada']].reset_index(drop=True)


# In[80]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
auxXPreco1 = pd.DataFrame(mms.fit_transform(df1[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']]))
auxXPreco2 = pd.DataFrame(mms.fit_transform(df2[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']]))
auxXPreco3 = pd.DataFrame(mms.fit_transform(df3[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']]))
auxXPreco4 = pd.DataFrame(mms.fit_transform(df4[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']]))


# In[81]:


auxXPreco1.columns = ['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']
auxXPreco2.columns = ['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']
auxXPreco3.columns = ['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']
auxXPreco4.columns = ['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']


# In[82]:


XPreco1[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
        'latitude', 'longitude']] = auxXPreco1[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']]
XPreco2[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
        'latitude', 'longitude']] = auxXPreco2[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']]
XPreco3[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
        'latitude', 'longitude']] = auxXPreco3[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']]
XPreco4[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
        'latitude', 'longitude']] = auxXPreco4[['area', 'quartos', 'vagasGaragem', 'andar', 'valor_condominio', 'menorDistanciaMetro',
                               'latitude', 'longitude']]


# In[83]:


XPreco1[['anoConstrucao','temBanheiro','temCozinha']] = XPreco1[['anoConstrucao','temBanheiro','temCozinha']].astype(str)
XPreco2[['anoConstrucao','temBanheiro','temCozinha']] = XPreco2[['anoConstrucao','temBanheiro','temCozinha']].astype(str)
XPreco3[['anoConstrucao','temBanheiro','temCozinha']] = XPreco3[['anoConstrucao','temBanheiro','temCozinha']].astype(str)
XPreco4[['anoConstrucao','temBanheiro','temCozinha']] = XPreco4[['anoConstrucao','temBanheiro','temCozinha']].astype(str)


# In[84]:


XPreco1 = pd.get_dummies(XPreco1)
XPreco2 = pd.get_dummies(XPreco2)
XPreco3 = pd.get_dummies(XPreco3)
XPreco4 = pd.get_dummies(XPreco4)


# In[88]:


from sklearn.model_selection import train_test_split
XPreco1_treino, XPreco1_teste,yPreco1_treino,yPreco1_teste = train_test_split(XPreco1,yPreco1,test_size=0.2, random_state=42)
XPreco2_treino, XPreco2_teste,yPreco2_treino,yPreco2_teste = train_test_split(XPreco2,yPreco2,test_size=0.2, random_state=42)
XPreco3_treino, XPreco3_teste,yPreco3_treino,yPreco3_teste = train_test_split(XPreco3,yPreco3,test_size=0.2, random_state=42)
XPreco4_treino, XPreco4_teste,yPreco4_treino,yPreco4_teste = train_test_split(XPreco4,yPreco4,test_size=0.2, random_state=42)


# In[90]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


# In[92]:


from catboost import CatBoostRegressor
modeloPreco1CB = CatBoostRegressor(iterations=1000,random_state=42)
modeloPreco1CB.fit(XPreco1_treino, yPreco1_treino)
modeloPreco2CB = CatBoostRegressor(iterations=1000,random_state=42)
modeloPreco2CB.fit(XPreco2_treino, yPreco2_treino)
modeloPreco3CB = CatBoostRegressor(iterations=1000,random_state=42)
modeloPreco3CB.fit(XPreco3_treino, yPreco3_treino)
modeloPreco4CB = CatBoostRegressor(iterations=1000,random_state=42)
modeloPreco4CB.fit(XPreco4_treino, yPreco4_treino)


# In[93]:


y_pred_Preco1CB = modeloPreco1CB.predict(XPreco1_teste)
print('Prazo1: ','MSE: ',mean_squared_error(yPreco1_teste, y_pred_Preco1CB), 'MAE: ',mean_absolute_error(yPreco1_teste,y_pred_Preco1CB),'MAPE: ',mean_absolute_percentage_error(yPreco1_teste,y_pred_Preco1CB),'R2: ',r2_score(yPreco1_teste, y_pred_Preco1CB))
y_pred_Preco2CB = modeloPreco2CB.predict(XPreco2_teste)
print('Prazo2: ','MSE: ',mean_squared_error(yPreco2_teste, y_pred_Preco2CB), 'MAE: ',mean_absolute_error(yPreco2_teste,y_pred_Preco2CB),'MAPE: ',mean_absolute_percentage_error(yPreco2_teste,y_pred_Preco2CB),'R2: ',r2_score(yPreco2_teste, y_pred_Preco2CB))
y_pred_Preco3CB = modeloPreco3CB.predict(XPreco3_teste)
print('Prazo3: ','MSE: ',mean_squared_error(yPreco3_teste, y_pred_Preco3CB), 'MAE: ',mean_absolute_error(yPreco3_teste,y_pred_Preco3CB),'MAPE: ',mean_absolute_percentage_error(yPreco3_teste,y_pred_Preco3CB),'R2: ',r2_score(yPreco3_teste, y_pred_Preco3CB))
y_pred_Preco4CB = modeloPreco4CB.predict(XPreco4_teste)
print('Prazo4: ','MSE: ',mean_squared_error(yPreco4_teste, y_pred_Preco4CB), 'MAE: ',mean_absolute_error(yPreco4_teste,y_pred_Preco4CB),'MAPE: ',mean_absolute_percentage_error(yPreco4_teste,y_pred_Preco4CB),'R2: ',r2_score(yPreco4_teste, y_pred_Preco4CB))


# In[94]:


pd.DataFrame(np.array(modeloPreco1CB.get_feature_importance(prettified=True))).head(50)


# In[3]:


get_ipython().system('pip install ipwidgets')


# In[1]:


import ipwidgets as widgets


# In[ ]:




