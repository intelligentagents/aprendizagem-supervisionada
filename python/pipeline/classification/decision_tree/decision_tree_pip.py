#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:50:14 2019

@author: r4ph
"""

# Importando os pacotes
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

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

# Gerando o pipeline de maneira mais simplificada:
# 1- Retira a primeira coluna; 
drop_column = FunctionTransformer(all_but_first_column, validate = False)
    
# 2- Completando os valores que faltam com a média: 
imputer = SimpleImputer(strategy='mean')

# 3- Criando o modelo de classificação
clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

# Criando a pipeline
pipe = make_pipeline(drop_column,imputer,clf)

# Definindo a validação cruzada com 5 folds:
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Treinando classificador com a validação cruzada
results = cross_validate(pipe, X, y, cv=kfold, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

# Visualizando os resultados 
pd.DataFrame.from_dict(results).mean()
