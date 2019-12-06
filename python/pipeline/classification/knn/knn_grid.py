# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando os pacotes
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Importando os dados
# Os dados contém informações relacionadas a empresas indianas coletadas por auditores com o objetivo de construir um modelo para realizar tarefas 
# de classificação de empresas suspeitas. Os atributos estão relacionados a métricas de auditorias como: scores, riscos, etc.
df = pd.read_csv('data/audit_risk.csv')

# Descrevendo o dataset
df.info()

df.describe()

# Visualizando o dataset
df.head(5)

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :17]
y = df.iloc[:, -1]

# Especificando as colunas numericas/categoricas: 
cat_columns = list(X.select_dtypes(include=['object']).columns)
num_columns = list(X.select_dtypes(include=np.number).columns)

# Dividindo o conjunto de treinamento e testes:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Criando uma pipeline com o search grid e column transform que consiste em transformar as colunas de acordo com os diferentes tipos:
# O parâmetro refit=True serve para retornar o modelo com os melhores parâmetros
model = make_pipeline(make_column_transformer((SimpleImputer(strategy='median', fill_value='missing'), num_columns),
                                              (OneHotEncoder(handle_unknown='ignore'), cat_columns)
                                              ),
                      GridSearchCV(KNeighborsClassifier(metric = 'minkowski', p = 2),
                                 param_grid={'n_neighbors': [3, 4, 5, 6, 7]},
                                 cv=5,
                                 refit=True))

#Treina o modelo com os melhores parâmetros
model.fit(X_train, y_train)

# Predizendo os valores do conjunto de testes:
y_pred = model.predict(X_test)

# Visualizando das métricas
print(classification_report(y_test, y_pred))
