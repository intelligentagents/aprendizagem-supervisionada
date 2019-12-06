# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:06:47 2019

@author: Jairo Souza
"""

# Importando os pacotes
from __future__ import absolute_import
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

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

# Dividindo o conjunto de treinamento e testes:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Criando uma pipeline com o search grid e column transform que consiste em transformar as colunas de acordo com os diferentes tipos:
model = make_pipeline(make_column_transformer((SimpleImputer(strategy='median', fill_value='missing'), num_columns),
                                       (StandardScaler(), num_columns),
                                       (OneHotEncoder(handle_unknown='ignore'), cat_columns)
                                       ),
                       GridSearchCV(LogisticRegression(random_state = 0),
                                 param_grid={'C': [1.0,2.0], '': ['l1','l2']},
                                 cv=5,
                                 refit=True)
                       )

#Treina o modelo com os melhores parâmetros
model.fit(X_train, y_train)

# Predizendo os valores do conjunto de testes:
y_pred = model.predict(X_test)

# Visualizando das métricas
print(classification_report(y_test, y_pred))
