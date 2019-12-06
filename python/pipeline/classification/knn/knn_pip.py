# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando os pacotes
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer

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

# Criando um preprocessador consiste em transformar as colunas de acordo com os diferentes tipos:
# Passos: 1- Insere a média dos valores numericos que estão faltando; 2- Cria as variáveis dummy para os valores categóricos
preprocessor = make_column_transformer((SimpleImputer(strategy='median', fill_value='missing'), num_columns),
                                       (OneHotEncoder(handle_unknown='ignore'), cat_columns)
                                       )

# Testando o preprocessador nas variáveis:
preprocessor.fit_transform(df).toarray()[:5]

# Criando uma pipeline com o preprocessador e o classificador:
model = make_pipeline(preprocessor,
                      KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))

#Mostrando os passos:
model.steps

# Treinando classificador com a validação cruzada
results = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

# Visualizando os resultados 
pd.DataFrame.from_dict(results).mean()
