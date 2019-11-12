# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando os pacotes
from __future__ import absolute_import
from utils import plot_results_class, accuracy, f_measure, feature_scaling
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

# Importando os dados
df = pd.read_csv('data/titanic.csv')

# Selecionando uma amostragem dos dados para melhor visualização
df = df.sample(n=100, random_state=0)

# Descrevendo o dataset
df.describe()

# Visualizando o dataset
df.head(5)

# Preenchendo os valores númericos nulos (NA) com a mediana.
df = df.fillna(df.median())

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, [5, 9]].values
y = df.iloc[:, 1].values

# Criando os subconjuntos de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalizando as features 
X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

# Treinando o modelo de Árvore de Decisão com o Conjunto de Treinamento
classifier =  RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Prevendo os resultados do modelo criado com o conjunto de testes
y_pred = classifier.predict(X_test)

# Criando a matriz de confusão com o conjunto de testes 
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Visualizando a métrica de acurácia através das funções criandas e da bibilioteca sklearn
accuracy(tp, fp, fn, tn)
classifier.score(X_test, y_test)

# Exibindo o f-measure
f_measure(tp, fp, fn)
f1_score(y_test, y_pred)  

# Exibindo os resultados do conjunto de treinamento
plot_results_class(X_train, y_train, classifier, 'Random Forest (Conj. de Treinamento)')

# Exibindo os resultados do conjunto de testes
plot_results_class(X_test, y_test, classifier, 'Random Forest (Conj. de Testes)')