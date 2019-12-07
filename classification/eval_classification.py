# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando os pacotes
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Importando os dados. O dataset consistem em conter caracteristicas de animais contendo atributos baseados em valores booleanos (0 ou 1).
# Portando, o objetivo é classificar qual tipo de animal (peixe, ave, etc.) de acordo com as caracteristicas deles (cabelo, penas, cauda, etc.).
df = pd.read_csv('data/zoo.csv')

# Descrevendo o dataset
df.info()

# Deletando as features que não tem importância no modelo: Nome
df = df.drop(['name'], axis = 1)

#Visualizando o dataset:
df.head(10)

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :16].values
y = df.iloc[:, 16].values

# Criando os subconjuntos de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Criando o dicionário contendo todos os classificadores
estimators = {'Decision Tree': DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
              'KNN': KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
              'Logistic Regression': LogisticRegression(random_state = 0),
              'Naive Bayes': GaussianNB(),
              'Random Forest': RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
              'SVC': SVC(kernel = 'rbf', random_state = 0)}


# Criando dataframe que irá guardar os resultados finais dos classificadores
df_results = pd.DataFrame(columns=['clf', 'acc', 'prec', 'rec', 'f1'], index=None)

# Itereando os classificadores
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

