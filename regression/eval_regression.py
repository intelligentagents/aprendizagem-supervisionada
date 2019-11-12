# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando as packages
from __future__ import absolute_import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# Importando os dados
df = pd.read_csv('data/pricing_houses.csv')

# Visualizando e descrevendo  o dataset
df.info()

df.head(5)

# Selecionando apenas as features numericas
df = df.select_dtypes(include=['int64', 'float64'])

# Descrevendo o dataset:
df.describe()

# Preenchendo os valores númericos nulos (NA) com a mediana.
df = df.fillna(df.median())

# Definindo as variáveis indepedentes e dependentes
X = df.iloc[:, :37].values
y = df.iloc[:, -1].values.reshape(-1,1)

# Dividindo o dataset em conjunto de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Normalização das features
# X_train = feature_scaling(X_train)
# X_test = feature_scaling(X_test)

# Criando o dicionário contendo todos os regressores
regressors = {'Linear Regression': LinearRegression(),
              'Decision Tree Reg:': DecisionTreeRegressor(random_state = 0),
              'Random Forest Reg': RandomForestRegressor(n_estimators = 10, random_state = 0),
              'SVR:': SVR(kernel = 'rbf')}

# Criando dataframe que irá guardar os resultados finais dos regressores
df_results = pd.DataFrame(columns=['reg', 'rmse', 'rmse_log', 'r_2_score'])

# Itereando os regressores
for name, reg in regressors.items():
    
    # Treinando os regressores com Conjunto de Treinamento
    reg.fit(X_train, y_train)
    
    # Prevendo os resultados com o conjunto de testes
    y_pred = reg.predict(X_test)
    
    df_results.loc[len(df_results), :] = [name, reg.score(X_test, y_test), 
                   mean_squared_error(y_test, y_pred), mean_squared_log_error(y_test, y_pred)]

# Exibindo os resultados:
df_results