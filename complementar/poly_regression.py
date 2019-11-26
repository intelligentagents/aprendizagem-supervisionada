# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:24:09 2019

@author: Jairo Souza
"""

# Importando as packages
from __future__ import absolute_import
from utils import feature_scaling
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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


# Normalização das features
X = feature_scaling(X)

# Treinando o modelo de regressão linear
regressor = LinearRegression()
regressor.fit(X, y)

# Transformando as features na ordem polinomial - 2
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

# Treinando o modelo de regressão polynomial
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Avaliando o modelo com a métrica r2
lin_reg_poly.score(X_poly, y)

# Prevendo os resultados 
y_pred = lin_reg_poly.predict(X_poly)