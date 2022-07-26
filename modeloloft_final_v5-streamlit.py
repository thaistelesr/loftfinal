
import streamlit as st
from fuzzywuzzy import fuzz

import pandas as pd
import numpy as np


#STREAMLIT

# Text/Title
st.title("PAUL | Modelo de Precificação de Venda de Imóvel")
st.subheader('Informações do apartamento')

with st.form(key='my_form_to_submit'):

    #Aplica streamlit | Input Usuario
    prazoVenda = st.number_input('Você pretende vender seu apartamento em até quantos dias?')

    cep = st.text_input("Digite somente os oito números do CEP:")
   

    numero = st.text_input("Digite o número do endereço:")


    area = st.text_input('Informe a área do imóvel (11-500 m²): ',0,500,10)


    andar = st.text_input('Informe o andar do imóvel (0-20):',0,20,0)


    quartos = st.selectbox('Informe o número de quartos:',['0','1','2','3','4','5','6'])
    

    banheiros = st.selectbox('Informe o número de banheiros:',['0','1','2','3','4','5','6'])
    

    vagasGaragem = st.selectbox('Informe a quantidade de vagas de garagem:', ['0','1','2','3','4','5','6','7'])
    


    valorCondominio = st.text_input('Informe o valor do condomínio:', 0,10000,0)

    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        
        if len(cep) > 0:
                if len(cep) != 8:
                    st.error("CEP Inválido")
                else:
                    st.success("CEP válido")
       
        if numero=='0':
            st.write("Informe um número válido")
        else:
            st.success("Número válido")
       
        if area=='0':
            st.write("Informe uma área válida")
        else:
            st.success("Área válida")
       
        if andar =='0':
            st.write("Digite um andar válido")
        else:
            st.success("Andar válido")

        st.write("Você informou",quartos,"quartos")
        st.write("Você informou",banheiros,"banheiros")
        st.write("Você informou",vagasGaragem,"vagas de gagarem")

        st.info("Aguarde! O Paul está calculando o preço do seu imóvel para o prazo informado!")


        #Import Base de Dados
        df = pd.read_csv('caracteristicas_aptos_final.csv')

        #Tratamento

        #Reordenando as colunas
        df = df[['id', 'city','latitude','longitude','area', 'floor','bedrooms','restrooms','parking_spots','price','created_at', 'status','valor_condominio', 
                    'is_published','last_unpublished_at', 'unpublish_reason', 'sold_at','building_id', 'unit_id']]

        #Renomeando as colunas
        df.rename(columns={'price':'preco','bedrooms':'quartos','parking_spots':'vagasGaragem', 'floor':'andar','created_at':'criadoEm','is_published':'publicado',
                                'last_unpublished_at':'despublicadoEm','unpublish_reason':'motivoDespublicacao','sold_at':'vendidoEm','building_id':'idPredio',
                                'unit_id':'idApartamento','city':'cidade','valor_condominio':'valorCondominio','restrooms':'banheiros'}, inplace=True)


        df[['criadoEm', 'despublicadoEm', 'vendidoEm']] = df[['criadoEm', 'despublicadoEm', 'vendidoEm']].apply(pd.to_datetime)

        #Filtrando o data frame para considerar dados da cidade de São Paulo e excluindo valores nulos
        df = df[(df.cidade == 'São Paulo') & (df.latitude.notna()) & (df.longitude.notna()) & (df.area.notna()) & (df.andar.notna()) & (df.quartos.notna()) 
                & (df.idPredio.notna()) & (df.idApartamento.notna()) & (df.vagasGaragem.notna()) & (df.preco.notna())  & (df.valorCondominio.notna()) & (df.cidade.notna()) 
                & (df.banheiros.notna())]
        df = df.drop('cidade',axis=1)

        df.drop_duplicates(subset=['id'],inplace=True)

        df['banheiros'] = df['banheiros'].astype(int)

        # eliminamos os apartamentos acima de 500m² e abaixo de 15m²
        i = df[df['area']>500].index
        j = df[df['area']<15].index
        df = df.drop(i)
        df = df.drop(j)

        # eliminamos os apartamentos acima de 3 milhões
        k = df[df['preco'] > 3_000_000].index
        df = df.drop(k)

        # Valores incoerentes nos apartarmentos com andar superior a 30. Por exemplo, 905 seria 9º andar, apto 905
        df['andar'] = df['andar'].apply(lambda x: x//10 if x>30 and x<300 else
                                                    x//100 if x>= 300 else
                                                    x)

        # eliminamos os apartamentos com condomínio acima de R$ 10.000
        l = df[df['valorCondominio'] > 3_000_000].index
        df = df.drop(l)

        # eliminamos apartamentos com mais de 8 vagas de garagem
        df = df[df['vagasGaragem'] <= 7]
        df = df[(df['banheiros'] > 0) & (df['banheiros'] <= 6)]


        df = df[(df['vendidoEm'].dt.year == 2021) | (df['vendidoEm'].dt.year == 2022)]
        df['tempoVenda'] = df['vendidoEm'] - df['criadoEm']
        df['tempoVenda'] = df['tempoVenda'].apply(lambda x: x.days)



        df['prazoVenda'] = df['criadoEm'] + pd.DateOffset(days=prazoVenda)
        min(df['criadoEm']),max(df['criadoEm']),max(df['despublicadoEm'][df['despublicadoEm'].notnull()]),max(df['vendidoEm'][df['vendidoEm'].notnull()])
        dataReferencia = max(df['criadoEm'])
        dataCorte = pd.to_datetime(dataReferencia) - pd.DateOffset(days=prazoVenda)
        df = df[df['criadoEm'] < dataCorte]
        df['situacao'] = df.apply(lambda x: 'vendido' if (x['status'] == 'SOLD' or x['motivoDespublicacao'] == 'UNIT_SOLD') and x['vendidoEm'] <= x['prazoVenda'] else 'excluir', axis=1)
        df = df[df.situacao != 'excluir']
        df = pd.get_dummies(df)



        #Treino do Modelo CatBoost

        y = df['preco'].reset_index(drop=True)
        X = df[['latitude', 'longitude', 'area', 'andar','quartos','banheiros','vagasGaragem','valorCondominio']].reset_index(drop=True)

        from sklearn.model_selection import train_test_split
        X_treino, X_teste,y_treino,y_teste = train_test_split(X,y,test_size=0.2, random_state=42)

        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler()
        X_treino = pd.DataFrame(mms.fit_transform(X_treino))
        X_teste =  pd.DataFrame(mms.transform(X_teste))
        X_treino.columns = ['latitude', 'longitude', 'area', 'andar','quartos','banheiros','vagasGaragem','valorCondominio']
        X_teste.columns = ['latitude', 'longitude', 'area', 'andar','quartos','banheiros','vagasGaragem','valorCondominio']

        from catboost import CatBoostRegressor
        modeloCB = CatBoostRegressor(random_state=42)
        modeloCB.fit(X_treino, y_treino)

        #aplicar streamlit

        #Exemplo input usuário
        #cep = '22011-002'
        #numero = 86
        #area = 38
        #andar = 1
        #quartos = 1
        #banheiros = 1
        #vagasGaragem = 0
        #valorCondominio = 600 


        # Transformação das features
        # CEP + Numero para Latitude/Longitude

        from pycep_correios import get_address_from_cep, WebService
        from geopy.geocoders import Nominatim

        try:
            endereco = get_address_from_cep(cep, webservice=WebService.APICEP)
        except:
            print('CEP_Inválido')

        geolocator = Nominatim(user_agent="test_app")
        local = geolocator.geocode(endereco['logradouro'] + ","+numero+" " + endereco['cidade'] + " - " + endereco['bairro'])


        exemplo = [local.latitude, local.longitude,area,andar,quartos,banheiros,vagasGaragem,valorCondominio]

        # Montagem do registro de entrada para aplicação da previsão
        exemplo = pd.DataFrame(exemplo).T
        exemplo.columns = ['latitude', 'longitude', 'area', 'andar', 'quartos', 'banheiros', 'vagasGaragem','valorCondominio']

        # Aplicação da função dummies para desmenbramento da variável andar
        #exemplo['andar'] = exemplo['andar'].apply(lambda x: 'baixo' if x <= 4 else 'intermediario' if x > 4 and x <= 10 else 'alto')
        #exemplo = pd.get_dummies(exemplo)
        #exemplo['andar_intermediario'] = 0
        #exemplo['andar_alto'] = 0
        #exemplo = exemplo[['latitude', 'longitude', 'area', 'andar_baixo', 'andar_intermediario','andar_alto','quartos','banheiros','vagasGaragem','valorCondominio']]

        # Aplicação do MinMaxScaler

        exemplo =  pd.DataFrame(mms.transform(exemplo))
        exemplo.columns = ['latitude', 'longitude', 'area', 'andar','quartos','banheiros','vagasGaragem','valorCondominio']


        #aplicar streamlit



        # Previsão
        preco = modeloCB.predict(exemplo)[0]
        preco = f'{preco:_.2f}'
        preco = preco.replace('.',',').replace('_','.')
        print(f'O valor estimado para venda em até {prazoVenda} dias é R$ {preco}')


        #Aplica streamlit | Input Usuario



        st.info("Preço calculado com sucesso!")
        st.write("O preço para seu imóvel, para o prazo de venda de ",prazoVenda," dias é de R$",preco)

