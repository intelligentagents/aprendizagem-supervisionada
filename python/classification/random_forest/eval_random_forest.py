# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:55:37 2019

@author: Jairo Souza
"""

# Importando os pacotes
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Importando os dados
# Os dados contém informações relacionadas a empresas indianas coletadas por auditores com o objetivo de construir um modelo para realizar tarefas 
# de classificação de empresas suspeitas. Os atributos estão relacionados a métricas de auditorias como: scores, riscos, etc.
df = pd.read_csv('data/audit_risk.csv')

# Descrevendo o dataset
df.info()

df.describe()

# Visualizando o dataset
df.head(5)

# Deletando coluna de localização:    
df = df.drop('LOCATION_ID', axis=1)

# Analisando se existem valores nulos:
df[df.isnull().values.any(axis=1)]

# Preechendo os valores nulos com a mediana:
df = df.fillna(df.median())

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :17].values
y = df.iloc[:, -1].values

# Criando os subconjuntos de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Criando o dicionário contendo todos os classificadores
estimators = {'Decision Tree': DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
              'KNN': KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
              'Random Forest': RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
              'SVC': SVC(kernel = 'rbf', random_state = 0)}


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