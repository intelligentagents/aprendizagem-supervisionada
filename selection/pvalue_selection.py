#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:46:53 2019

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
# Portanto, a cada iteração é adicionada uma feature que deverá ser analisada seu impacto no modelo através do *p-value*.

# Analisando os coeficientes com a package OLS:
# Selecionando apenas a primeira feature:
X_opt = X_train[:, [0,1]]
regressor_ols = sm.OLS(endog = y_train, exog = X_train[:, [0,1]]).fit()
regressor_ols.summary()


# Selecionando apenas as features de indice 0-const, 1-LotArea, 2-PoolArea
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

# Analisando os valores acima, vimos que todas as features possuem um P-value significativo, ou seja, dentro do intervalo definido (SL = .05) .
# Portanto, não iremos adicionar ela ao modelo final, visto que ela não impacta de maneira positiva ao modelo. 
# Finalmente, esse ciclo se repete até que todas as features sejam analisadas.

# Treinando o modelo de regressão com o conjunto de treinamento
regressor = LinearRegression()
regressor.fit(X_test[:, [1,2,3]], y_test)

# Calculando a acurácia do modelo:
regressor.score(X_test[:, [1,2,3]], y_test)


################### Backward Elimination: ###################
# Realizando o processo de Backward Elimination. Esse processo é realizado através de uma análise da contribuição de todas as features ao modelo final. 
# Portanto, a cada iteração é removida uma feature que deverá ser analisada seu impacto no modelo através do *p-value*.

# Analisando todas as features no modelo:
X_opt = X_train[:, [0,1,2,3,4,5,6,7]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Analisando todas as features, exceto 6-YrSold
X_opt = X_train[:, [0,1,2,3,4,5,7]]
regressor_ols = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_ols.summary()

# Analisando os valores acima, vimos que todas as features possuem um valor muito próximo de zero.
# Portanto, seguindo o processo de Backward Selection, todas as features acima devem ser mantidas no model.

# Treinando o modelo de Regressão com todas as features, exceto X6
regressor = LinearRegression()
regressor.fit(X_train[:, [1,2,3,4,5,7]], y_train)

# Analisando o novo score do modelo com a métrica r²:
regressor.score(X_test[:, [1,2,3,4,5,7]], y_test)

