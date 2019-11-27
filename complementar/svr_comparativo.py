# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:08:02 2019

@author: Jairo Souza
"""

# Importando as packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from utils import  feature_scaling
from sklearn.metrics import mean_squared_error, mean_squared_log_error


# Importando o dataset do nosso estudo. Esse dataset consiste em prever a qualidade de vinhos tintos (entre 0 a 10). 
# Portanto, queremos prever a qualidade de vinho tintos através de atributos como: PH do vinho, acidez do vinho, etc.
df = pd.read_csv('data/wine_quality_red.csv', sep = ";")

# Exporando o dataset
df.info()

# Visualizando o sumário das colunas numéricas do dataset
df.describe()

#Visualizando dados:
df.head(10)

# Analisando se algumas colunas do atribuito horsepower contém valores nulos:
df[df.isnull().values.any(axis=1)]

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :11].values
y = df.iloc[:, -1].values.reshape(-1,1)

# Criando o dicionário com os regressores:
regressors = {'Linear Regression': LinearRegression(), 'SVR:': SVR(kernel = 'rbf')}

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Criando o dataframe que irá armazenar o resultado final dos regressores:
df_results = pd.DataFrame(columns=['reg', 'r_2_score', 'rmse'])

# Normalização das features
X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

# Percorrendo o dicionário e treinando e avaliando os modelos:
for name, reg in regressors.items():
    
    # Treinando os regressores com Conjunto de Treinamento
    reg.fit(X_train, y_train)
    
    # Prevendo os resultados com o conjunto de testes
    y_pred = reg.predict(X_test)
    
    df_results.loc[len(df_results), :] = [name, reg.score(X_test, y_test), 
                   mean_squared_error(y_test, y_pred)]
    
# Exibindo os resultados:
df_results    