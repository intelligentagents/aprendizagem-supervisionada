# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:27:30 2019

    Script responsável pelo pré-processamento dos dados.
    
"""

# Importing the libraries
from __future__ import absolute_import
import pandas as pd
from utils import k_fold_cv, feature_scaling, accuracy, precision, recall, informedness, markdness
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# Importando o Dataset
df = pd.read_csv('data/preprocessing_data.csv')

# Exporando o dataset
df.info()

df.describe()

df.head(5)

# Visualizando a distribuição das features
df.boxplot()

df.hist()

# Preenchendo os registros nuḿericos que não possuem valores com a media , mediana ou zeros.

df.fillna(df.mean())

df.fillna(df.median())

df.fillna(0)

# Detectando registros com valores nulos no dataframe:
df[df.isnull().any(axis=1)]

# Deletando registros com valores nulos:
df.dropna()

# Preenchendo com a mediana
df = df.fillna(df.median())

# Definindo as variáveis dependentes e independentes
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

# Codificando os valores das variáveis dependente (y) com valores númericos.
le = LabelEncoder()

# Ajustando e transformando o valor de y.
y = df.iloc[:, 4].values
y = le.fit_transform(y)

# Criando variaǘeis Dummy
# Obs: Algumas técnicas de aprendizado de máquina exigem que você elimine uma dimensão da apresentação 
# para evitar a dependência entre as variáveis. Isso pode ser feito através do parâmetro "drop_first = True".
X = pd.get_dummies(df.iloc[:, :-1] ,prefix=['city', 'sex'], drop_first=True).values
y = df.iloc[:, 4].values

# Aplicando a normalização/escalonamento das features
X = feature_scaling(X)


# Definindo os valores da matriz de confusão:
tp, fp, fn, tn = [1,1,1,2]

# Calculando os valores da acurácia
accuracy (tp, fp, fn, tn)

# Calculando os valores da precisão
precision (tp, fp)

# Calculando os valores do recall
recall(tp, fn)

# Calculando os valores do informedness 
informedness(tp, fp, fn, tn)

# Calculando os valores do markdness
markdness(tp, fp, fn, tn)


# Usando a função para definir os indices de uma validação cruzada com 5 folds.
k_fold_cv(list(df.index.values))
 

# Calidação cruzada com 5 folds usando o sklearn
kf = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in kf.split(X):
    print("Indices de Treinamento:", train_index, "Indices de Testes:", test_index)
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]


# Dividindo o dataset no conjunto de treinamento (80%) e testes (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

