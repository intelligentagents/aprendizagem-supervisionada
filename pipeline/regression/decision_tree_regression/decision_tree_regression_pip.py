# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:31:48 2019

@author: Jairo Souza
"""

# Importando as packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:,1:8]
y = df.iloc[:,0]

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Criando uma pipeline com o preprocessador e o classificador:
# Passos: 1- Insere a média dos valores numericos que estão faltando; 2- Normaliza dos dados númericos;
model = make_pipeline(SimpleImputer(strategy='median', fill_value='missing'),
                      StandardScaler(),
                      DecisionTreeRegressor(random_state = 0)
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