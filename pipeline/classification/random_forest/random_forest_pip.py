# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:37:56 2019

@author: Jairo Souza
"""

# Importando os pacotes
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

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

# Verificando os tipos das colunas:
df.info()

# Definindo as variáveis dependentes/independentes.
X = df[df.columns[~df.columns.isin(['status'])]].values
y = df['status'].values.reshape(-1,1)

# Criando uma pipeline com o preprocessador e o classificador:
# Passos: 1- Insere a média dos valores numericos que estão faltando; 2- Normaliza dos dados númericos;
model = make_pipeline(SimpleImputer(strategy='median', fill_value='missing'),
                      StandardScaler(),
                      RandomForestClassifier(criterion = 'entropy', random_state = 0, n_estimators = 10))

#Mostrando os passos:
model.steps

# Treinando classificador com a validação cruzada
results = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

# Visualizando os resultados 
pd.DataFrame.from_dict(results).mean()