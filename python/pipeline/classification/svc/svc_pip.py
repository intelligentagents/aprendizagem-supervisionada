# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:51:14 2019

@author: Jairo Souza
"""

# Importando os pacotes
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from sklearn.datasets import load_breast_cancer

# Importando os dados
df = load_breast_cancer()

#Visualizando as features do dataset:
df.feature_names

#Visualizando dados:
df.data[0:5, ]

# Definindo as variáveis dependentes/independentes.
X = df.data
y = df.target

# Criando uma pipeline com o preprocessador e o classificador:
# Passos: 1- Insere a média dos valores numericos que estão faltando; 2- Normaliza dos dados númericos;
model = make_pipeline(SimpleImputer(strategy='median', fill_value='missing'),
                      StandardScaler(),
                      SVC(kernel = 'rbf', random_state = 0))

#Mostrando os passos:
model.steps

# Treinando classificador com a validação cruzada
results = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

# Visualizando os resultados 
pd.DataFrame.from_dict(results).mean()