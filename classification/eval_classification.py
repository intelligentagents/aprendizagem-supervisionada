# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando os pacotes
from __future__ import absolute_import
from utils import accuracy, precision, recall, f_measure, informedness, markdness, feature_scaling
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Importando os dados
df = pd.read_csv('data/titanic.csv')

# Descrevendo o dataset
df.info()

# Visualizando o dataset
df.head(5)

# Deletando as features que não tem importância no modelo: Nome, Código do Ticket e Código da Cabine:
df = df.drop(['Name','Ticket','Cabin', 'PassengerId'], axis = 1)

# Preenchendo os valores númericos nulos (NA) com a mediana.
df = df.fillna(df.median())

# Criando variaǘeis Dummy nas variáveis categóricas
df = pd.get_dummies(df ,prefix=['Sex', 'Embarked'], drop_first=True)

#Visualizando o dataset tratado:
df.head(10)

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Criando os subconjuntos de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalizando as features 
X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

# Criando o dicionário contendo todos os classificadores
estimators = {'Decision Tree': DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
              'KNN': KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
              'Logistic Regression': LogisticRegression(random_state = 0),
              'Naive Bayes': GaussianNB(),
              'Random Forest': RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
              'SVC': SVC(kernel = 'rbf', random_state = 0)}



# Criando dataframe que irá guardar os resultados finais dos classificadores
df_results = pd.DataFrame(columns=['clf', 'acc', 'prec', 'rec', 'f1', 'inform', 'mark'], index=None)

# Itereando os classificadores
for name, estim in estimators.items():
    
    # print("Treinando Estimador {0}: ".format(name))
    
    # Treinando os classificadores com Conjunto de Treinamento
    estim.fit(X_train, y_train)
    
    # Prevendo os resultados do modelo criado com o conjunto de testes
    y_pred = estim.predict(X_test)
    
    # Criando a matriz de confusão com o conjunto de testes 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # print("accuracy {0}: ".format(accuracy(tp, fp, fn, tn)))
    
    # Armazenando as métricas de cada classificador em um dataframe
    df_results.loc[len(df_results), :] = [name, accuracy(tp, fp, fn, tn), precision (tp, fp), recall(tp, fn), 
                   f_measure(tp, fp, fn), informedness(tp, fp, fn, tn), markdness(tp, fp, fn, tn)]

# Exibindo os resultados finais
df_results

