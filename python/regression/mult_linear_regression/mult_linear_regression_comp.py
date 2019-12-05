# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:18:13 2019

@author: Jairo Souza
"""

# Importando as packages
from __future__ import absolute_import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Normalização das features
# X_train = feature_scaling(X_train)
# X_test = feature_scaling(X_test)

# Treinando o modelo de regressão linear multipla com o conjunto de treinamento
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Avaliando o modelo com a métrica r2
regressor.score(X_test, y_test)

# Prevendo os resultados com o conjunto de testes
y_pred = regressor.predict(X_test)
