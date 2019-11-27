# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando as packages
from __future__ import absolute_import
from utils import feature_scaling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

# Importando os dados
# O dataset contém dados gerais e preços das casas de boston. O objetivo é predizer o valor das casas.
boston = load_boston()

#Transformando os dados em um dataframe
df = pd.DataFrame(boston.data, columns = boston.feature_names)

#Adicionando o valor do preço das casas (target) ao dataframe:
df['PRICE'] = boston.target

# Visualizando e descrevendo  o dataset
df.info()

df.head(5)

# Descrevendo o dataset:
df.describe()


# Definindo as variáveis indepedentes e dependentes
X = df.iloc[:, :13].values
y = df.iloc[:, -1].values

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Normalização das features
X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

# Criando o dicionário contendo todos os regressores
regressors = {'Linear Regression': LinearRegression(),
              'Decision Tree Reg:': DecisionTreeRegressor(random_state = 0),
              'Random Forest Reg': RandomForestRegressor(n_estimators = 10, random_state = 0),
              'SVR:': SVR(kernel = 'rbf')}

# Criando dataframe que irá guardar os resultados finais dos regressores
df_results = pd.DataFrame(columns=['reg', 'r_2_score', 'rmse'])

# Itereando os regressores
for name, reg in regressors.items():
    
    # Treinando os regressores com Conjunto de Treinamento
    reg.fit(X_train, y_train)
    
    # Prevendo os resultados com o conjunto de testes
    y_pred = reg.predict(X_test)
    
    df_results.loc[len(df_results), :] = [name, reg.score(X_test, y_test), 
                   mean_squared_error(y_test, y_pred)]

# Exibindo os resultados:
df_results