#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:08:59 2019

@author: r4ph
"""

# Importando os pacotes
from __future__ import absolute_import
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Importando os dados
# O objetivo é determinar se uma pessoa ganhar mais de 50k por ano.
df = pd.read_csv('data/adult.csv')

# Visualizando o dataset
df.head(5)

#Descrevendo o dataset:
df.info()

# Transformando as colunas categóricas em numericas:
# Selecionando as colunas:
columns = df.select_dtypes(include=['object']).columns

# Criando o labelEncoder
le =  LabelEncoder()

for column in columns:
    df[column] = le.fit_transform(df[column])


# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :14].values
y = df.iloc[:, -1].values

# Criando os subconjuntos de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Treinando o modelo de Regressão Logistica com o Conjunto de Treinamento
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Prevendo os resultados do modelo criado com o conjunto de testes
y_pred = classifier.predict(X_test)

# Exibindo a matriz de confusão com o conjunto de testes 
confusion_matrix(y_test, y_pred)

# Criando dataframe que irá guardar os resultados finais:
df_results = pd.DataFrame(columns=['clf', 'acc', 'prec', 'rec', 'f1'], index=None)

# Armazenando as métricas em um dataframe:
df_results.loc[len(df_results), :] = ['Regressão Logística', accuracy_score(y_test, y_pred), precision_score (y_test, y_pred, average = 'macro'),
                   recall_score(y_test, y_pred,  average = 'macro'), f1_score(y_test, y_pred,  average = 'macro')]

df_results