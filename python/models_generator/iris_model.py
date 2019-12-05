# Importando Bibliotecas
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn import datasets

#Definindo o Path:
PATH = '/home/r4ph/desenv/projetos/aprendizagem-supervisionada/ml-apps/models_generator'

# Função que salva o modelo no disco:
def save_model(model, model_name = 'classifier.joblib'):
    joblib.dump(model, os.path.join(PATH, model_name))

# Função que importa o modelo:
def import_model(model_name):
    return joblib.load(os.path.join(PATH, model_name))

# Get the dataset
df = datasets.load_iris()

# Divide o dataset nas variáveis dependentes e independentes
X = df.data
y = df.target

# Divide o dataset no conjunto de treinamento (80%) e testes (20%):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)

# Construção do modelo de classificação com o comjunto de treinamento
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
# Predição no conjunto de testes:
prediction = classifier.predict(X_test)

# Visualização da matriz de confusão:
print("Confusion Matrix:")
print(confusion_matrix(y_test, prediction))

# Salva o modelo:
save_model(classifier)

# Importa o modelo salvo:
model = import_model('classifier.joblib')

# Utiliza o modelo importado para fazer uma nova predição:
# Criando um dado qualquer
instance = np.array([4.9, 3. , 1.4, 0.2]).reshape(1, -1)

# Definindo os labels:
types = { 0: "Iris Setosa", 1: "Iris Versicolour ", 2: "Iris Virginica"}

# Criando a predição e retornando o valor das labels:
prediction = model.predict(instance)
types[prediction[0]]



