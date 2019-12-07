# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:16:48 2019

@author: Jairo Souza
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

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
y = df['status'].values

# Dividindo o conjunto de treinamento e testes:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando uma pipeline com o preprocessador e o classificador:
# Passos: 1- Insere a média dos valores numericos que estão faltando; 2- Normaliza dos dados númericos;
model = make_pipeline(SimpleImputer(strategy='median', fill_value='missing'),
                      StandardScaler(),
                      GridSearchCV(RandomForestClassifier(criterion = 'entropy', random_state = 0),
                                 param_grid={'n_estimators': [8,9,10], 'min_samples_split': [2,3]},
                                 cv=5,
                                 refit=True)
                      )

#Mostrando os passos:
model.steps

#Treina o modelo com os melhores parâmetros
model.fit(X_train, y_train)

# Predizendo os valores do conjunto de testes:
y_pred = model.predict(X_test)

# Visualizando das métricas
print(classification_report(y_test, y_pred))