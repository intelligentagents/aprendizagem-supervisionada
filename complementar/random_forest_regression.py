# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:25:43 2019

@author: Jairo Souza
"""

# Importando as packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils import  feature_scaling
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Importando o dataset do nosso estudo. 
# Esta é uma tarefa dificil de regressão, em que o objetivo é prever a área queimada de incêndios florestais em Portugal usando dados meteorológicos
df = pd.read_csv('data/forestfires.csv')

# Exporando o dataset
df.info()

# Visualizando o sumário das colunas numéricas do dataset
df.describe()

#Visualizando dados:
df.head(10)

# Codificando os valores das variáveis categóricas com valores númericos.
le = LabelEncoder() 
df['month'] =  le.fit_transform(df['month'])
df['day'] =  le.fit_transform(df['day'])

#Visualizando novamente os dados após as transformações: 
df.head(10)

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:,:12].values
y = df.iloc[:,12].values

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Normalização das features
X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

# Treinando o modelo de regressão com o conjunto de treinamento
regressor =  RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Prevendo os resultados com o conjunto de testes
y_pred = regressor.predict(X_test)

# Avaliando o modelo com a métrica r2
regressor.score(X_test, y_test)

# Avaliando o modelo com a métrica rmse
mean_squared_error(y_test, y_pred)

