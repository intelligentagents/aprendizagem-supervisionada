#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:47:12 2019

@author: r4ph
"""

# Importando libs
from __future__ import absolute_import
from utils import feature_scaling
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.regression.linear_model as sm

#Importando o dataset.
df = pd.read_csv('data/pricing_houses.csv')

#Selecionando algumas features dos dados para uma melhor visualização do problema
df = df.loc[:, ['LotArea', 'PoolArea', 'GarageArea', 'OverallCond','YearBuilt', 'YrSold', 'Fireplaces',
                'SalePrice']]

# Visualizando o dataset:
df.describe()

# Preenchendo os valores númericos nulos (NA) com a mediana.
df = df.fillna(df.median())

df.head(5)

# Definindo as variáveis dependentes/independentes
X = df[df.columns[~df.columns.isin(['SalePrice'])]].values
y = df['SalePrice'].values.reshape(-1,1)

# Normalização das features:
X = feature_scaling(X)

# Inserindo uma coluna preenchida com valores 1 no começo da matriz de feature para que seja realizado os cálculos necessários.
X = np.append(arr = np.ones((1460,1)).astype(int), values = X, axis =1)

# Dividindo os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train[1:5,:5]

################### Foward Elimination: ###################
# Esse processo é realizado através de uma análise incremental da contribuição das features ao modelo final. 
# Portanto, a cada iteração é adicionada uma feature que deverá ser analisada seu impacto no modelo através da métrica r²

# Utilizando a package OLS para calculo dos coeficientes:
# Selecionando apenas a primeira feature:
X_opt = X_train[:, [0,1]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Selecionando apenas as features de indice 0-Constante, 1-LotArea, 2-PoolArea
X_opt = X_train[:, [0,1,2]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Selecionando apenas as features de indice 0-const, 1-LotArea, 2-PoolArea, 3-GarageArea 
X_opt = X_train[:, [0,1,2,3]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Selecionando apenas as features de indice 0-const, 1-LotArea, 2-PoolArea, 3-GarageArea, 4-OverallCond
X_opt = X_train[:, [0,1,2,3,4]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Selecionando apenas as features de indice 0-const, 1-LotArea, 2-PoolArea, 3-GarageArea, 5-YearBuilt
X_opt = X_train[:, [0,1,2,3,5]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Selecionando apenas as features de indice 0-const, 1-LotArea, 2-PoolArea, 3-GarageArea, 5-YearBuilt, 6-YearSold
X_opt = X_train[:, [0,1,2,3,5,6]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Selecionando apenas as features de indice 0-const, 1-LotArea, 2-PoolArea, 3-GarageArea, 5-YearBuilt, 7-Fireplaces
X_opt = X_train[:, [0,1,2,3,5,7]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Após analise da combinação das features acima, vimos que o vetor de features ([1,2,3,5,7]) é o que mais contribui para 
# o modelo. Portanto, o processo é finalizado, visto que já temos uma lista suficiente de features que impactam de maneira positiva o modelo de Regressão (aumento do R²).

# Treinando o modelo com as features selecionadas e com o conjunto de treinamento:
regressor = LinearRegression()
regressor.fit(X_train[:, [1,2,3,5,7]], y_train)

# Analisando o score do modelo com a métrica R² no conjunto de testes:
regressor.score(X_test[:, [1,2,3,5,7]], y_test)


################### Backward Elimination: ###################
# Esse processo é realizado através de uma análise inversao ao modelo Foward. 
#Portanto, a cada iteração é removida uma feature que deverá ser analisada seu impacto no modelo através da métrica R².

# Analisando todas as features:
X_opt = X_train[:, [0,1,2,3,4,5,6,7]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

#Analisando todas as features, exceto a feature 6-YrSold
X_opt = X_train[:, [0,1,2,3,4,5,7]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

#Analisando todas as features, exceto a feature 6-YrSold e 7-Fireplaces
X_opt = X_train[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

#Analisando todas as features, exceto a 5-YearBuilt, 6-YrSold e 7-Fireplaces
X_opt = X_train[:, [0,1,2,3,4]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Analisando o valor do Adjusted R² acima, vimos que o valor diminuiu consideravelmente após a retirada da 
# feature Ano de Construção (*YearBuilt*):. Portanto, tal feature não deve ser retirada. 
# Desse modo, podemos finalizar o processo e utilizar o modelo com o maior R² (contendo essa feature).


# Treinando o modelo com as features selecionadas no conjunto de treinamento:
regressor = LinearRegression()
regressor.fit(X_train[:, [1,2,3,4,5,7]], y_train)

# Analisando o novo score do modelo no conjunto de testes com a métrica R²:
regressor.score(X_test[:, [1,2,3,4,5,7]], y_test)