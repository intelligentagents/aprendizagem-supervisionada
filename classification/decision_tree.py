# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

"""

# Importando os pacotes
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

# Importando os dados
df = pd.read_csv('data/glass.csv')

# Descrevendo o dataset
df.describe()

# Visualizando o dataset
df.head(5)

# Definindo as variáveis dependentes/independentes.
X = df.iloc[:, :10].values
y = df.iloc[:, -1].values



# 1ª Maneira de trreinamento e avaliação
#Definindo as métricas a serem utilizadas:
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Treinando o modelo de Árvore de Decisão usando a validação cruzada com 10 folds:
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

scores = cross_validate(classifier, X, y, cv=10, scoring=metrics)

# Visualizando as métricas
pd.DataFrame.from_dict(scores)

pd.DataFrame.from_dict(scores).mean()


# 2ª Maneira de trreinamento e avaliação
# Defininndo o numero de folds: 5. Lembrando que também iremos realizar um "shuffle" nos dados:
kf = KFold(n_splits=5, shuffle= True)

#Define o modelo a ser treinado 
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

# Criando dataframe que irá guardar os resultados finais das métricas
df_results = pd.DataFrame(columns=['itera', 'acc', 'prec', 'rec', 'f1'], index=None)

#Realizando o treinamento do classificador usando validação cruzada:

itera = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
 
    # Treina o modelo a cada iteração com o conjunto de treinamento:
    classifier.fit(X_train, y_train)

    # Prevendo os resultados do modelo criado com o conjunto de testes
    y_pred = classifier.predict(X_test)
       
    # Armazenando os valores das métricas a cada iteração com o conjunto de teste em um df
    itera += 1
    df_results.loc[len(df_results), :] = [itera, accuracy_score(y_test, y_pred), precision_score (y_test, y_pred, average = 'macro'),
                   recall_score(y_test, y_pred,  average = 'macro'), f1_score(y_test, y_pred,  average = 'macro')]
    
#Criando um dataframe contendoo valor das métricas a cada iteração:
df_results

#Exibindo a media das metricas:
df_results.mean()