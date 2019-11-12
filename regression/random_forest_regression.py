# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando as packages
from __future__ import absolute_import
from utils import plot_results_reg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Importando os dados
df = pd.read_csv('data/pricing_houses.csv')

# Selecionando uma amostragem dos dados para melhor visualização
df = df.loc[:, ['LotArea', 'PoolArea', 'GarageArea', 'OverallCond','YearBuilt', 'MSZoning', 'SalePrice']].sample(n=60, random_state=0, weights = 'SalePrice')
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
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Avaliando o modelo com a métrica r2
regressor.score(X_test, y_test)

# Prevendo os resultados com o conjunto de testes
y_pred = regressor.predict(X_test)

# Visualizando os resultados do conjunto de treinamento
plot_results_reg(X_train, y_train, regressor, 'Random Forest Regression (Conj. de Treinamento)')

# Visualizando os resultados do conjunto de testes
plot_results_reg(X_test, y_test, regressor, 'Random Forest Regression (Conj. de Testes)')