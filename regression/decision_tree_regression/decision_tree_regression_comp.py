# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:25:43 2019

@author: Jairo Souza
"""

# Importando as packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from utils import  feature_scaling
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
# Exibindo algumas das linhas tinham valores nulos via indíces:
df.iloc[[32,126,374],]

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:,1:8]
y = df.iloc[:,0]

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Normalização das features
X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

# Treinando o modelo de regressão com o conjunto de treinamento
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Prevendo os resultados com o conjunto de testes
y_pred = regressor.predict(X_test)

# Avaliando o modelo com a métrica r2
regressor.score(X_test, y_test)

# Avaliando o modelo com a métrica rmse
mean_squared_error(y_test, y_pred)