# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:45:00 2019

Classificação de Imagens usando Árvore de Decisão

@author: Jairo Souza
"""
# Importando os pacotes

from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt

def plot_mnist (X, index):
    some_digit_image = X[index].reshape(64, 64)
    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
    interpolation="nearest")
    plt.axis("off")
    plt.show()

# Importando os dados - Dataset MNIST
# Fonte: https://www.openml.org/d/554
# mnist_data = fetch_openml('mnist_784')

mnist_data = load_digits()

# https://www.openml.org/d/554
mnist_data.data.shape

# Definindo as variáveis :
X, y, images = mnist_data["data"], mnist_data["target"], mnist_data["images"]

# Os dados são feitos de imagens 8x8 de dígitos.  Observe que cada imagem tem o mesmo tamanho e que as features são compostas de valores entre 0 e 15 que 
# consiste na escala de cinza de cada pixel na imagen, assim totalizando 64 features - 8x8:
X[0]

y[0]

# Visualizando os valores e imagem de digito qualquer usando o matplotlib
plt.gray() 
plt.matshow(mnist_data.images[0]) 
plt.show()

# Create a classifier: a support vector classifier
classifier = SVC(gamma=0.001)

# Dividindo os dados:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o classificador com o conjunto de treinamento:
classifier.fit(X_train, y_train)

# Predizendo os valores dos cojuntos de teses
y_pred = classifier.predict(X_test)

# Relatório das metricas de classificação:
print("Relatório de Classificação do SVC: %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))
    
# Matriz de Confusão:
metrics.confusion_matrix(y_test, y_pred)
