# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando as packages
from __future__ import absolute_import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm

# Importando os dados
df = pd.read_csv('data/pricing_houses.csv')
df = df.loc[:, ['LotArea', 'PoolArea', 'GarageArea', 'OverallCond','YearBuilt', 'MSZoning', 'SalePrice']].sample(n=60, random_state=0, weights = 'SalePrice')
# df.to_csv('data/pricing_houses_small.csv')

# Visualizando e descrevendo  o dataset
df.describe()

df.head(5)

# Codificando as variáveis Categóricos e evitando a armadilha da variável Dummy
df = pd.get_dummies(df , columns = ['MSZoning'], drop_first=True)

# Definindo as variáveis indepedentes e dependentes
X = df[df.columns[~df.columns.isin(['SalePrice'])]].values
y = df['SalePrice'].values.reshape(-1,1)

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

# Backaward Elimination:
X = np.append(arr = np.ones((60,1)).astype(int), values = X, axis =1)
X_opt = X[:, [0,1,2,3,4,5,6,8]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()