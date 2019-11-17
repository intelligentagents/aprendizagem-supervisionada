# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:37:54 2019

Análise de Sentimentos em dados do Twitter sobre o Governo de Minas.

@author: Jairo Souza
"""
# Importando os pacotes
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split


# Importanto os dados
# Fonte: https://github.com/minerandodados/mdrepo
df = pd.read_csv('data/tweets.csv', encoding ='utf-8')

df.head(5)

df.describe()
# Visualizando o dataset
sns.countplot(x= 'Classificacao',data = df)

# Por img no NOtebook
# https://miro.medium.com/max/1726/1*CfcTH_TIWNqVDDWI94aK5g.png


X = df['Text'].values
y = df['Classificacao'].values

# Criação de um vetor que irá calcular a frequencia de todas as palavras 
vectorizer = CountVectorizer(ngram_range=(1,2))

# Converter o texto em uma matriz de contagens de tokens
freq_tweets = vectorizer.fit_transform(X)

# Dividindo os dados:
X_train, X_test, y_train, y_test = train_test_split(freq_tweets, y, test_size=0.20, random_state=42)

# Create the model
model = MultinomialNB()
model.fit(X_train, y_train)

results = cross_val_predict(model, X_test, y_test, cv=10)

metrics.accuracy_score(y_test, results)

print(metrics.classification_report(y_test,results,['Positivo','Negativo','Neutro']))