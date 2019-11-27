# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:37:56 2019

@author: Jairo Souza
"""

# Importando os pacotes
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

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

# Definindo as métricas a serem utilizadas:
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Treinando o modelo de Classificação usando a validação cruzada com 10 folds:
classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0, n_estimators = 10)
scores = cross_validate(classifier, X, y, cv=10, scoring=metrics)

# Visualizando os resultados finais:
pd.DataFrame.from_dict(scores)

# Exibindo os valores da media das métricas:
pd.DataFrame.from_dict(scores).mean()