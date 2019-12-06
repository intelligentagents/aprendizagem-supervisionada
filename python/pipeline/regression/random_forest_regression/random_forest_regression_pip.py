# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:31:48 2019

@author: Jairo Souza
"""

# Importando as packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import make_column_transformer

# Importando o dataset do nosso estudo. 
# Esta é uma tarefa dificil de regressão, em que o objetivo é prever a área queimada de incêndios florestais em Portugal usando dados meteorológicos
df = pd.read_csv('data/forestfires.csv')

# Exporando o dataset
df.info()

# Visualizando o sumário das colunas numéricas do dataset
df.describe()

#Visualizando dados:
df.head(10)

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:,:12]
y = df.iloc[:,12]

# Especificando as colunas numericas/categoricas: 
cat_columns = list(X.select_dtypes(include=['object']).columns)
num_columns = list(X.select_dtypes(include=np.number).columns)

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Criando um preprocessador consiste em transformar as colunas de acordo com os diferentes tipos:
# Passos: 1- Insere a média dos valores numericos que estão faltando; 2- Normaliza dos dados númericos; 3- Cria as variáveis dummy para os valores categóricos
model = make_pipeline(make_column_transformer((SimpleImputer(strategy='median', fill_value='missing'), num_columns),
                                              (StandardScaler(), num_columns),
                                              (OneHotEncoder(handle_unknown='ignore'), cat_columns)
                                              ),
                      RandomForestRegressor(n_estimators = 10, random_state = 0)
                      )

#Mostrando os passos:
model.steps

#Treinando o modelo:
model.fit(X_train,y_train)

# Prevendo os resultados com o conjunto de testes
y_pred = model.predict(X_test)

# Avaliando o modelo com a métrica r2
model.score(X_test, y_test)

# Avaliando o modelo com a métrica rmse
mean_squared_error(y_test, y_pred)

