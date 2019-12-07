#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:08:59 2019

@author: r4ph
"""

# Importando os pacotes
from __future__ import absolute_import
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

# Importando os dados
# O objetivo é determinar se uma pessoa ganhar mais de 50k por ano.
df = pd.read_csv('data/adult.csv')

# Visualizando o dataset
df.head(5)

#Descrevendo o dataset:
df.info()

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :14]
y = df.iloc[:, -1]

# Especificando as colunas numericas/categoricas: 
cat_columns = list(X.select_dtypes(include=['object']).columns)
num_columns = list(X.select_dtypes(include=np.number).columns)

# Criando um preprocessador consiste em transformar as colunas de acordo com os diferentes tipos:
# Passos: 1- Insere a média dos valores numericos que estão faltando; 2- Normaliza dos dados númericos; 3- Cria as variáveis dummy para os valores categóricos
preprocessor = make_column_transformer((SimpleImputer(strategy='median', fill_value='missing'), num_columns),
                                       (StandardScaler(), num_columns),
                                       (OneHotEncoder(handle_unknown='ignore'), cat_columns)
                                       )

# Testando o preprocessador nas variáveis:
preprocessor.fit_transform(df).toarray()[:5]

# Criando uma pipeline com o preprocessador e o classificador:
model = make_pipeline(preprocessor,
                      LogisticRegression(random_state = 0))

#Mostrando os passos:
model.steps

# Treinando classificador com a validação cruzada
results = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

# Visualizando os resultados 
pd.DataFrame.from_dict(results).mean()