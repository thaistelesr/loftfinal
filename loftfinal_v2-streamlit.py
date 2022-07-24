#!/usr/bin/env python
# coding: utf-8

# In[77]:


import streamlit as st
import string
from fuzzywuzzy import fuzz


# In[78]:


import pandas as pd
import numpy as np


# In[81]:


df = pd.read_csv('caracteristicas_aptos_final.csv')


# In[4]:


df.head()


# In[5]:


#df.columns


# In[6]:


df = df[['id', 'city','latitude','longitude','area', 'floor','bedrooms','restrooms','parking_spots','price','created_at', 'status','valor_condominio', 
            'is_published','last_unpublished_at', 'unpublish_reason', 'sold_at','building_id', 'unit_id']]


# In[7]:


df.rename(columns={'price':'preco','bedrooms':'quartos','parking_spots':'vagasGaragem', 'floor':'andar','created_at':'criadoEm','is_published':'publicado',
                         'last_unpublished_at':'despublicadoEm','unpublish_reason':'motivoDespublicacao','sold_at':'vendidoEm','building_id':'idPredio',
                         'unit_id':'idApartamento','city':'cidade','valor_condominio':'valorCondominio','restrooms':'banheiros'}, inplace=True)
df.head()


# In[8]:


df.shape


# In[9]:


df.isna().sum()


# In[10]:


df = df[(df.cidade == 'São Paulo') & (df.latitude.notna()) & (df.longitude.notna()) & (df.area.notna()) & (df.andar.notna()) & (df.quartos.notna()) 
        & (df.idPredio.notna()) & (df.idApartamento.notna()) & (df.vagasGaragem.notna()) & (df.preco.notna())  & (df.valorCondominio.notna()) & (df.cidade.notna()) 
        & (df.banheiros.notna())]
df = df.drop('cidade',axis=1)


# In[11]:


df.isna().sum()


# In[12]:


df.shape


# In[13]:


df.drop_duplicates(subset=['id'],inplace=True)
df.shape


# In[14]:


df.describe()


# In[15]:


df['area'].nlargest(10),df['area'].nsmallest(10) 


# In[16]:


df['area'].hist(bins=[0,50,100,150,200,300,400,500,1000])


# In[17]:


df['area'].describe()


# In[18]:


df['classe_area'] = df['area'].apply(lambda x: 'pequeno' if x <= 10 else 'intermediario' if x > 15 and x <= 100 else 'grande' if x > 80 and x <= 400 else 'gigante')
df['classe_area'].value_counts()


# In[19]:


df = df[(df['area'] >=10) | (df['area'] <= 500)]
df.shape


# In[20]:


round(df['preco'].nlargest(100)), round(df['preco'].nsmallest(100))


# In[21]:


df['preco'].hist(bins=[0, 250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000, 2250000, 2500000, 2750000, 3000000])


# In[22]:


df = df[(df['preco']>=50000) & (df['preco']<3000000)]
df.shape


# In[23]:


df['andar'].nlargest(30)


# In[24]:


df['andar'].hist(bins=[0,5,10,15,20,30,40])


# In[25]:


df = df[df['andar'] < 20]
df['andar'] = df['andar'].apply(lambda x: 'baixo' if x <= 4 else 'intermediario' if x > 4 and x <= 10 else 'alto')
df.shape


# In[26]:


df['andar'].value_counts()


# In[27]:


df['valorCondominio'].nlargest(20)


# In[28]:


df = df[df['valorCondominio'] <= 10000]
df.shape


# In[29]:


df['vagasGaragem'].value_counts()


# In[30]:


df = df[df['vagasGaragem'] <= 7]
df.shape


# In[31]:


df['quartos'].value_counts()


# In[32]:


df['banheiros'].value_counts()


# In[33]:


df = df[df['banheiros'] <= 6]
df.shape


# In[34]:


df['status'].unique()


# In[35]:


df['motivoDespublicacao'].unique()


# In[36]:


df['motivoDespublicacao'].value_counts()


# In[37]:


df[['criadoEm','despublicadoEm','vendidoEm']] = df[['criadoEm','despublicadoEm','vendidoEm']].apply(pd.to_datetime)


# In[38]:


df['criadoEm'].dt.year.value_counts()


# In[39]:


df['vendidoEm'].dt.year.value_counts()


# In[40]:


df['vendidoEm'][df['vendidoEm'].dt.year == 2020].dt.month.hist(bins=12,figsize=(8,8))


# In[41]:


df['vendidoEm'][df['vendidoEm'].dt.year == 2021].dt.month.hist(bins=12,figsize=(8,8))


# In[42]:


df['vendidoEm'][df['vendidoEm'].dt.year == 2022].dt.month.hist(bins=7,figsize=(8,8))


# In[43]:


df = df[(df['vendidoEm'].dt.year == 2021) | (df['vendidoEm'].dt.year == 2022)]


# In[44]:


df['tempoVenda'] = df['vendidoEm'] - df['criadoEm']


# In[45]:


df['tempoVenda'] = df['tempoVenda'].apply(lambda x: x.days)


# In[46]:


df['tempoVenda'].hist(bins=range(0,1000,30))


# In[47]:


df['tempoVenda'][df.tempoVenda >= 360].count()


# In[48]:


df['tempoVenda'][df.tempoVenda <= 360].count()


# In[49]:


df = df[df['tempoVenda'] <= 360]
df.shape


# In[50]:


prazoVenda = int(input('Você pretende vender seu apartamento em até quantos dias?'))


# In[51]:


df['prazoVenda'] = df['criadoEm'] + pd.DateOffset(days=prazoVenda)


# In[52]:


df[['criadoEm','prazoVenda']].head()


# In[53]:


df['area'].hist(bins=[0,30,50,75,100,150,200,250,300])


# In[54]:


min(df['criadoEm']),max(df['criadoEm']),max(df['despublicadoEm'][df['despublicadoEm'].notnull()]),max(df['vendidoEm'][df['vendidoEm'].notnull()])


# In[55]:


dataReferencia = max(df['criadoEm'])
dataCorte = pd.to_datetime(dataReferencia) - pd.DateOffset(days=prazoVenda)


# In[56]:


df = df[df['criadoEm'] < dataCorte]


# In[57]:


df['situacao'] = df.apply(lambda x: 'vendido' if (x['status'] == 'SOLD' or x['motivoDespublicacao'] == 'UNIT_SOLD') and x['vendidoEm'] <= x['prazoVenda'] else 'excluir', axis=1)


# In[58]:


df['situacao'].value_counts()


# In[59]:


df = df[df.situacao != 'excluir']


# In[60]:


df['situacao'].value_counts()


# In[61]:


df.columns


# In[62]:


df = pd.get_dummies(df)


# In[63]:


y = df['preco'].reset_index(drop=True)
X = df[['latitude', 'longitude', 'area', 'andar_baixo', 'andar_intermediario','andar_alto','quartos','banheiros','vagasGaragem','valorCondominio']].reset_index(drop=True)


# In[64]:


X.info(),type(y[0])


# In[65]:


from sklearn.model_selection import train_test_split
X_treino, X_teste,y_treino,y_teste = train_test_split(X,y,test_size=0.2, random_state=42)


# In[66]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_treino = pd.DataFrame(mms.fit_transform(X_treino))
X_teste =  pd.DataFrame(mms.transform(X_teste))
X_treino.columns = ['latitude', 'longitude', 'area', 'andar_baixo', 'andar_intermediario','andar_alto','quartos','banheiros','vagasGaragem','valorCondominio']
X_teste.columns = ['latitude', 'longitude', 'area', 'andar_baixo', 'andar_intermediario','andar_alto','quartos','banheiros','vagasGaragem','valorCondominio']


# In[67]:


from catboost import CatBoostRegressor
modeloCB = CatBoostRegressor(random_state=42)
modeloCB.fit(X_treino, y_treino)


# In[68]:


from sklearn.ensemble import RandomForestRegressor
modeloRF = RandomForestRegressor(random_state=42)
modeloRF.fit(X_treino, y_treino)


# In[69]:


import xgboost as xgb
modeloXGB = xgb.XGBRegressor(n_estimators = 1000, random_state = 42)
modeloXGB.fit(X_treino, y_treino)


# In[ ]:


from sklearn.linear_model import LinearRegression
modeloRL = LinearRegression()
modeloRL.fit(X_treino, y_treino)


# In[ ]:


from sklearn.metrics import mean_absolute_percentage_error, r2_score
y_predCB = modeloCB.predict(X_teste)
resultadoCB = pd.DataFrame([mean_absolute_percentage_error(y_teste,y_predCB).round(4),r2_score(y_teste, y_predCB).round(4)]).T

y_predRF = modeloRF.predict(X_teste)
resultadoRF = pd.DataFrame([mean_absolute_percentage_error(y_teste,y_predRF).round(4),r2_score(y_teste, y_predRF).round(4)]).T

y_predXGB = modeloXGB.predict(X_teste)
resultadoXGB = pd.DataFrame([mean_absolute_percentage_error(y_teste,y_predXGB).round(4),r2_score(y_teste, y_predXGB).round(4)]).T

y_predRL = modeloRL.predict(X_teste)
resultadoRL = pd.DataFrame([mean_absolute_percentage_error(y_teste,y_predRL).round(4),r2_score(y_teste, y_predRL).round(4)]).T

resultado = pd.concat([resultadoCB,resultadoRF,resultadoXGB,resultadoRL])
resultado.columns = ['MAPE','R2']
resultado.index = ['catboost','random_forrest','xgboost','regressao_linear']
resultado


# In[ ]:


fi = pd.DataFrame(np.array(modeloCB.get_feature_importance(prettified=True)))
fi.columns = ['feature','importancia']
fi


# In[ ]:


X_teste.columns


# In[ ]:


# Exemplo input usuário
cep = '22011-002'
numero = 86
area = 38
andar = 1
quartos = 1
banheiros = 1
vagasGaragem = 0
valorCondominio = 600 


# In[ ]:


# Transformação das features
# CEP + Numero para Latitude/Longitude

from pycep_correios import get_address_from_cep, WebService
from geopy.geocoders import Nominatim
try:
    endereco = get_address_from_cep(cep, webservice=WebService.APICEP)
except:
    print('CEP_Inválido')

geolocator = Nominatim(user_agent="test_app")
local = geolocator.geocode(endereco['logradouro'] + ","+str(numero)+" " + endereco['cidade'] + " - " + endereco['bairro'])


exemplo = [local.latitude, local.longitude,area,andar,quartos,banheiros,vagasGaragem,valorCondominio]
exemplo


# In[ ]:


# Montagem do registro de entrada para aplicação da previsão


exemplo = pd.DataFrame(exemplo).T
exemplo.columns = ['latitude', 'longitude', 'area', 'andar', 'quartos', 'banheiros', 'vagasGaragem','valorCondominio']

# Aplicação da função dummies para desmenbramento da variável andar
exemplo['andar'] = exemplo['andar'].apply(lambda x: 'baixo' if x <= 4 else 'intermediario' if x > 4 and x <= 10 else 'alto')
exemplo = pd.get_dummies(exemplo)
exemplo['andar_intermediario'] = 0
exemplo['andar_alto'] = 0
exemplo = exemplo[['latitude', 'longitude', 'area', 'andar_baixo', 'andar_intermediario','andar_alto','quartos','banheiros','vagasGaragem','valorCondominio']]

# Aplicação do MinMaxScaler
exemplo =  pd.DataFrame(mms.transform(exemplo))
exemplo.columns = ['latitude', 'longitude', 'area', 'andar_baixo', 'andar_intermediario','andar_alto','quartos','banheiros','vagasGaragem','valorCondominio']

exemplo


# In[ ]:


# Previsão

preco = modeloCB.predict(exemplo)[0]
preco = f'{preco:_.2f}'
preco = preco.replace('.',',').replace('_','.')
print(f'O valor estimado para venda em até {prazoVenda} dias é R$ {preco}')


# In[ ]:


#Título
st.write("""
PAUL: Modelo de Precificação de Venda de Imóvel\n
App que utiliza machine learning para prever o melhor preço de venda de imóvel de acordo com o prazo estimado pelo proprietário\n
Fonte: Dados Loft
""")


# In[ ]:


#Cabeçalho
st.subheader('Informações do proprietário')


# In[ ]:


#Nome do usuário
user_input = st.sidebar.text_input('Digite seu nome')
st.write('Proprietário: ', user_input)


# In[76]:


#Dados dos usuários com a função
def get_user_data():
    valor_cond = st.sidebar.slider('valorCondominio', 0,10000,0)
    tamanho = st.sidebar.slider('area',10,500,10)
    bedrooms = st.sidebar.slider('quartos',0,6,1)
    floor = st.sidebar.slider('andar',0,20,0)
    restrooms = st.sidebar.slider('banheiros',0,6,0)
    parking = st.sidebar.slider('vagasGaragem', 0,7,0)
    
    user_data = {'valorCondominio': valor_cond,
                 'area': tamanho,
                 'quartos':bedrooms,
                 'andar':floor,
                 'banheiros': restrooms,
                 'vagasGaragem': parking}
    
    features = pd.DataFrame(user_data, index = [0])
    
    return features

user_input_variables = get_user_data()

