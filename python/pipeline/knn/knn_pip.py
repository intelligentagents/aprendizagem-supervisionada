# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando os pacotes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

# Importando os dados
# Os dados contém informações relacionadas a empresas indianas coletadas por auditores com o objetivo de construir um modelo para realizar tarefas 
# de classificação de empresas suspeitas. Os atributos estão relacionados a métricas de auditorias como: scores, riscos, etc.
df = pd.read_csv('data/audit_risk.csv')

# Descrevendo o dataset
df.info()

df.describe()

# Visualizando o dataset
df.head(5)

# Analisando se existem valores nulos:
df[df.isnull().values.any(axis=1)]

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :17].values
y = df.iloc[:, -1].values

# Criando os subconjuntos de treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Definindo as métricas a serem utilizadas:
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Criando o modelo usando pipeline:
# Etapas: 1- Transforma as colunas categóricas em valores numericos; 2- completa os valores que faltam com a média ; 3- Cria o modelo
model = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('one-hot encoder', OneHotEncoder()),
    ('knn', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))
])

#Definindo as métricas a serem utilizadas:
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Definindo a validação cruzada com 5 folds:
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Treinando classificador com a validação cruzada
results = cross_validate(model, X, y, cv=kfold, scoring=metrics)

# Visualizando das métricas
print("Average accuracy (std): %f (%f)" %(results['test_accuracy'].mean(), results['test_accuracy'].std()))

# # Visualizando das métricas por iteração:
pd.DataFrame.from_dict(results)

# Visualizando das métricas pela média
pd.DataFrame.from_dict(results).mean()
