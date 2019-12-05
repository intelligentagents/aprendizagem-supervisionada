# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importing the packages
from __future__ import absolute_import
from utils import plot_results_linear, plot_results_poly
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importando os dados
df = pd.read_csv('data/pricing_houses.csv')

# Selecionando uma amostragem dos dados para melhor visualização
df = df.loc[:, ['LotArea', 'PoolArea', 'GarageArea', 'OverallCond','YearBuilt', 'MSZoning', 'SalePrice']].sample(n=30, random_state=0, weights = 'SalePrice')
# df.to_csv('data/pricing_houses_small.csv')

# Visualizando e descrevendo  o dataset
df.describe()

df.head(5)

# Definindo as variáveis indepedentes e dependentes
X = df.loc[:, 'LotArea'].values.reshape(-1,1)
y = df.loc[:, 'SalePrice'].values.reshape(-1,1)

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Normalização das features
# X_train = feature_scaling(X_train)
 #X_test = feature_scaling(X_test)

# Treinando o modelo de regressão linear com o conjunto de treinamento
regressor = LinearRegression()
regressor.fit(X, y)

# Transformando as features na ordem polinomial - 4
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

# Treinando o modelo de regressão polynomial com o conjunto de treinamento
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Avaliando o modelo com a métrica r2
lin_reg_poly.score(X_poly, y)

# Prevendo os resultados com o conjunto de testes
y_pred = lin_reg_poly.predict(X_poly)

# Comprando os resultados entre a regressão linear e polinomial
# Visualizando os resultados da regressão linear:
plot_results_linear(X, y, regressor, 'Regressão Linear')

# Visualizando os resultados da regressão linear polinomial:
plot_results_poly(X, y, lin_reg_poly, poly_reg, 'Regressão Polynomial')