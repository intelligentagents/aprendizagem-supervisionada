# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 01:05:46 2019

@author: Jairo Souza
"""

# Importando as packages
from __future__ import absolute_import
from utils import feature_scaling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Importando o dataset do nosso estudo. O ojetivo é prever o consumo médio de carros através da coluna mpg (galões de combustível por milhas)
# Portanto, queremos prever o grau de economia de cada modelo de carro através de atributos como: número de cilindros, peso, potência, etc..
df = pd.read_csv('data/cars.csv', sep = ";")

# Exporando o dataset
df.info()

# Visualizando o sumário das colunas numéricas do dataset
df.describe()

#Visualizando dados:
df.head(10)

# Analisando se algumas colunas do atribuito horsepower contém valores nulos:
df[df.isnull().values.any(axis=1)]

# Preechendo os valores nulas com a mediana
df = df.fillna(df.median())


# Definindo as variáveis dependentes/independentes.
X = df.iloc[:,1:8]
y = df.iloc[:,0]

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Normalização das features
X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

# Criando o dicionário contendo todos os regressores
regressors = {'Linear Regression': LinearRegression(),
              'Decision Tree Reg:': DecisionTreeRegressor(random_state = 0),
              'Random Forest Reg': RandomForestRegressor(n_estimators = 10, random_state = 0),
              'SVR:': SVR(kernel = 'rbf')}

# Criando dataframe que irá guardar os resultados finais dos regressores
df_results = pd.DataFrame(columns=['reg', 'r_2_score', 'rmse'])

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