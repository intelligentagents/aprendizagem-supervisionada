# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:51:14 2019

@author: Jairo Souza
"""

# Importando os pacotes
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

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

# Dividindo o conjunto de treinamento e testes:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando uma pipeline com o preprocessador e o classificador:
# Passos: 1- Insere a média dos valores numericos que estão faltando; 2- Normaliza dos dados númericos;
model = make_pipeline(SimpleImputer(strategy='median', fill_value='missing'),
                      StandardScaler(),
                      GridSearchCV(SVC(gamma = 'auto'),
                                 param_grid={'C': [1.0,1.1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                                 cv=5,
                                 refit=True)
                      )

#Mostrando os passos:
model.steps

#Treina o modelo com os melhores parâmetros
model.fit(X_train, y_train)

# Predizendo os valores do conjunto de testes:
y_pred = model.predict(X_test)

# Visualizando das métricas
print(classification_report(y_test, y_pred))