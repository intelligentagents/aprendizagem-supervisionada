# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:31:48 2019

@author: Jairo Souza
"""

# Importando as packages
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_diabetes

# Importando os dados
# Dez variáveis  relacionadas a  idade, sexo, índice de massa corporal, pressão arterial média 
#e seis medidas séricas foram obtidas para cada um dos  442 pacientes com diabetes,

df = load_diabetes()

#Visualizando as features do dataset:
df.feature_names

#Visualizando dados:
df.data[0:5, ]

# Definindo as variáveis dependentes/independentes.
X = df.data
y = df.target

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Criando uma pipeline com o preprocessador e o classificador:
# Passos: 1- Insere a média dos valores numericos que estão faltando; 2- Normaliza dos dados númericos;
model = make_pipeline(SimpleImputer(strategy='median', fill_value='missing'),
                      StandardScaler(),
                      LinearRegression()
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

