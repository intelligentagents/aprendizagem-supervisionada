#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:50:14 2019

@author: r4ph
"""

# Importando os pacotes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Função que retorna todas as colunas, exceto a primeira:
def all_but_first_column(X):
    return X[:, 1:]

# Importando os dados
# Os dados são contém atributos de vidros. Portanto, o objetivo é classificar corretamente os tipos de vidros (Vidro de carro, Prédios, etc.) 
df = pd.read_csv('data/glass.csv')

# Descrevendo o dataset
df.describe()

# Visualizando o dataset
df.head(5)


# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :10].values
y = df.iloc[:, -1].values

# Dividindo o conjunto de treinamento e testes:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1ª Maneira: https://scikit-learn.org/stable/_images/grid_search_cross_validation.png
# Criando o modelo usando pipeline:
# Etapas: 1- Retira a primeira coluna; 2- completa os valores que faltam com a média ; 3- Cria o modelo
pipe = Pipeline(steps=[
    ('drop id column', FunctionTransformer(all_but_first_column, validate = False)),
    ('imputer', SimpleImputer(strategy='mean')),
    ('tree', DecisionTreeClassifier(max_depth=3, random_state=0))
])


 # Tunando os hiperparâmetros:
parameters = {'tree__max_depth': [3, 4, 5]}

# Definindo a validação cruzada com 5 folds:
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Usando a busca em grid na pipeline:
grid = GridSearchCV(pipe, param_grid=parameters, cv=kfold, n_jobs=-1)

# Treinando o modelo com o conjunto de treinamento:
grid.fit(X_train, y_train)

# Analisando qual o melhor parâmetro:
grid.best_params_

# Retreinando o classificador comos melhores hiperparâmetros usando a validação cruzada
grid.best_estimator_.fit(X_train, y_train)

# Predizendo os valores do conjunto de testes:
y_pred = grid.best_estimator_.predict(X_test)

# Visualizando das métricas
print(classification_report(y_test, y_pred))

# 2ª Maneira:
# Criando o modelo usando pipeline:
# Etapas: 1- Retira a primeira coluna; 2- completa os valores que faltam com a média ; 3- Cria o modelo com o grid search
model = make_pipeline(FunctionTransformer(all_but_first_column, validate = False),
                      SimpleImputer(strategy='mean'),
                      GridSearchCV(DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
                                 param_grid={'max_depth': [3, 4, 5]},
                                 cv=5,
                                 refit=True))

#Treina o modelo:
model.fit(X_train, y_train)

# Predizendo os valores do conjunto de testes:
y_pred = model.predict(X_test)

# Visualizando das métricas
print(classification_report(y_test, y_pred))
