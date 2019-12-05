# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:51:14 2019

@author: Jairo Souza
"""

# Importando os pacotes
from __future__ import absolute_import
from utils import plot_results_class, accuracy, f_measure, feature_scaling
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from sklearn.datasets import load_breast_cancer

# Importando os dados
df = load_breast_cancer()

# Selecionando uma amostragem dos dados para melhor visualização
# df = df.sample(n=100, random_state=0)

#Visualizando as features do dataset:
df.feature_names

#Visualizando dados:
df.data[0:5, ]

# Definindo as variáveis dependentes/independentes.
X = df.data
y = df.target

# Criando os subconjuntos de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalizando as features 
X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

# Treinando o modelo de SVC com o Conjunto de Treinamento
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Prevendo os resultados do modelo criado com o conjunto de testes
y_pred = classifier.predict(X_test)

# Criando a matriz de confusão com o conjunto de testes 
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Visualizando a métrica de acurácia através das funções criandas e da bibilioteca sklearn
# accuracy(tp, fp, fn, tn)
classifier.score(X_test, y_test)

# Exibindo o f-measure
f_measure(tp, fp, fn)