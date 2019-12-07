# -*- coding: utf-8 -*-

# Importando as packages
from __future__ import absolute_import
from utils import feature_scaling
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error


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

# Treinando o modelo de Regressão Múltipla:
regressor = LinearRegression()
regressor.fit(X, y)

# Transformando as features na 2ª ordem polinomial:
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

# Retreinando o modelo de Regressão Polinomial:
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Criando dataframe que irá guardar os resultados finais dos regressores
df_results = pd.DataFrame(columns=['Name','r_2_score', 'rmse'])

# Inserindo as métricas de Regressão Múltipla e Regressão Polinomial no dataframe:
# Regressão Múltipla
y_pred = regressor.predict(X)

df_results.loc[len(df_results), :] = ['Reg. Múltipla', regressor.score(X, y), mean_squared_error(y, y_pred)]

#Regressão Polynomial
y_pred = lin_reg_poly.predict(X_poly)

df_results.loc[len(df_results), :] = ['Reg Polinomial', lin_reg_poly.score(X_poly, y), mean_squared_error(y, y_pred)]

# Exibindo os resultados:
df_results