#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:09:44 2019

@author: r4ph
"""

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Importando os dados
# O conjunto de dados inclui informações demográficas, hábitos e registros médicos históricos de 858 pacientes.
# O objetivo é predizer se um determinado paciente tem indicadores de câncer cervical.
df = pd.read_csv('data/risk_factors_cervical_cancer.csv')

# Descrevendo o dataset
df.info()

# Visualizando o dataset
df.head(5)

#Deletando colunas relacionadas a timestamp, visto que estão praticamente nulas:
df = df.drop(['STDs_ Time since first diagnosis', 'STDs_ Time since last diagnosis'], axis = 1)

# Substituindo valores com ? por NaN:
df = df.replace('?', np.NaN)

# Transformando todas as colunas em númericas:
df = df.apply(pd.to_numeric)

# Analisando valores nulos:
df[df.isnull().values.any(axis=1)]

# Preenchendo os valores númericos nulos (?) com a mediana.
df = df.fillna(df.mean())

# Visualizando o dataset após as transformações:
df.head(5)

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :35].values
y = df.iloc[:, -1].values

# Criando os subconjuntos de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Criando o dicionário contendo todos os classificadores
estimators = {'Decision Tree': DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
              'KNN': KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean'),
              'SVC': SVC(kernel = 'rbf', random_state = 0),
              'Random Forest' : RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0) ,
              'Naive Bayes' : GaussianNB(),
              'Logistic Regression' : LogisticRegression(random_state = 0)}


# Criando dataframe que irá guardar os resultados finais dos classificadores
df_results = pd.DataFrame(columns=['clf', 'acc', 'prec', 'rec', 'f1'], index=None)

# Percorrendo os classificadores
for name, estim in estimators.items():
    
    # print("Treinando Estimador {0}: ".format(name))
    
    # Treinando os classificadores com Conjunto de Treinamento
    estim.fit(X_train, y_train)
    
    # Prevendo os resultados do modelo criado com o conjunto de testes
    y_pred = estim.predict(X_test)
    
    
    # Armazenando as métricas de cada classificador em um dataframe
    df_results.loc[len(df_results), :] = [name, accuracy_score(y_test, y_pred), precision_score (y_test, y_pred, average = 'macro'),
                   recall_score(y_test, y_pred,  average = 'macro'), f1_score(y_test, y_pred,  average = 'macro')]

# Exibindo os resultados finais
df_results