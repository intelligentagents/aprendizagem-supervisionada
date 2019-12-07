# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 01:22:36 2019

@author: Jairo Souza
"""

# Importando os pacotes
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics

# Naive Bayes com labes binários
# O dataset contém atributos de clima, temp, humidade, etc. Portanto, o objetivo é calcular a probabilidade de jogar de acordo com os atributos clima e tempo.
df = pd.read_csv('data/tennis.csv')

# Selecionando apenas alguns atributos para melhor visualização:
df = df.loc[:, ['wheather', 'temp', 'play']]

# Descrevendo o dataset
df.describe()

df.info()

# Visualizando o dataset
df.head(5)

# Criando o labelEncoder
le = preprocessing.LabelEncoder()

# Transformando os dados categóricos em numericos:
df['wheather'] = le.fit_transform(df['wheather'])

# Similarmente, transformando as outras colunas categóricas em numericas:
df['temp'] = le.fit_transform(df['temp'])
df['play'] = le.fit_transform(df['play'])

#Combinando as features (wheather e temp) em uma unica variável:
#df['weather_temp'] = zip(df['weather'].values, df['temp'].values)

#Definindo as variáveis dependentes/independentes:

X = df.iloc[:, :2].values 
y = df.iloc[:, -1].values

# Criando um classificador Naive Bayes e treinando:
model = GaussianNB()
model.fit(X,y)

# Prevendo os resultados:
predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild

print("Valor a ser previsto:", predicted)


# Naive Bayes com labels multiplos
# Carregando dataset que consistem em uma base de dados contendo atributos químicos de vinhos:
wine = datasets.load_wine()

# Visualizando o nome das features:
print ("Features: ", wine.feature_names)

# Visualizando os labels (tipos de vinhos)
print ("Labels: ", wine.target_names)

# Visualizando a formato dos dados:
wine.data.shape

#Exibindo os 5 primeiros registros:
wine.data[0:5]

# Transformando os dados em um dataframe:
df_wine = pd.DataFrame(wine.data, columns = wine.feature_names)

#Adicionando os labels ao dataframe:
df_wine['type'] = wine.target

#Visualizando o dataframe:
df_wine.head(5)

#Definindo as variáveis dependentes/independentes:
X = df_wine.iloc[:, :13]
y = df_wine.iloc[:, -1]

# Dividindo os daods em conjunto de treinamento/testes (70% treinamento / 30% testes)
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=42) 

# Criando um classificador Naive Bayes e treinando:
model = GaussianNB()
model.fit(X_train,y_train)

# Predizendo as respostas para o conjunto de testes
y_pred = model.predict(X_test)

# Visualizando a acurácia do modelo:
print("Acurácia:", metrics.accuracy_score(y_test, y_pred))

