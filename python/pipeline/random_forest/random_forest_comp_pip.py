# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:37:56 2019

@author: Jairo Souza
"""

# Importando os pacotes
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# Esse conjunto de dados é composto por uma série de medidas biomédicas de 31 pacientes, sendo 23 com doença de Parkinson (DP). 
# Cada coluna  é uma medida de voz específica e cada linha corresponde a uma das 195 gravações de voz desses indivíduos.
# O principal objetivo dos dados é discriminar pessoas saudáveis daquelas com Parkinson de acordo com a coluna "status", que é definida como 0 para saudável e 1 para DP.
df = pd.read_csv('data/parkinsons.csv')

# Descrevendo o dataset
df.describe()

# Visualizando o dataset
df.head(5)

# Retirando a coluna com o nome do dataframe:
df = df.drop(['name'], axis=1)

# Definindo as variáveis dependentes/independentes.
X = df[df.columns[~df.columns.isin(['status'])]].values
y = df['status'].values.reshape(-1,1)

# Definindo as métricas a serem utilizadas:
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Treinando o modelo de Classificação usando a validação cruzada com 10 folds:
classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0, n_estimators = 10)
scores = cross_validate(classifier, X, y, cv=10, scoring=metrics)

# Visualizando os resultados finais:
pd.DataFrame.from_dict(scores)

# Exibindo os valores da media das métricas:
pd.DataFrame.from_dict(scores).mean()