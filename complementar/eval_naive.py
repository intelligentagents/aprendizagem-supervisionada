#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:55:54 2019

@author: r4ph
"""

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Importando os dados
# O dataset contém atributos de clima, temp, humidade, etc. Portanto, o objetivo é calcular a probabilidade de jogar de acordo com os atributos clima e tempo.
df = pd.read_csv('data/tennis.csv')

# Descrevendo o dataset
df.describe()

# Visualizando o dataset
df.head(5)

# Transformando as colunas categóricas em numericas:
# Criando o labelEncoder
le =  LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Visualizando o dataset após as transformações:
df.head(5)

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :10].values
y = df.iloc[:, -1].values

# Criando os subconjuntos de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Criando o dicionário contendo todos os classificadores
estimators = {'Decision Tree': DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
              'KNN': KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean'),
              'SVC': SVC(kernel = 'rbf', random_state = 0),
              'Random Forest' : RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0) ,
              'Naive Bayes' : GaussianNB()}


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
