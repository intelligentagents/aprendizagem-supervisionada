# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:10:19 2019

@author: Jairo Souza
"""

# Instalando libs:
# %pip install mtcnn
# %pip install tensorflow
# %pip install keras

#Importando pacotes:
import os
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from numpy import savez_compressed
from keras.models import load_model
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

# Função que retorna a featurização das faces 
def get_embedding(model, face_pixels):
	# Escala de pixels:
	face_pixels = face_pixels.astype('float32')
	# Padronizando o valor dos pixels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transforma a face em uma amostra
	samples = expand_dims(face_pixels, axis=0)
	# retorna a featurização da amostra através do modelo
	yhat = model.predict(samples)
	return yhat[0]

# Função que carega as faces de um diretório:
def load_faces(directory):
	faces = list()
	# enumerate arquivos
	for filename in os.listdir(directory):
		# caminho
		path = directory + filename
		# extração da imagem
		face = extract_face(path)
		# armazenamento
		faces.append(face)
	return faces

# Extrai uam face através de uma foto
def extract_face(filename, required_size=(160, 160)):
	# carrega imagem 
	image = Image.open(filename)
	# conversão para RGB 
	image = image.convert('RGB')
	# conversão para um array
	pixels = asarray(image)
	# cria o detector usando os pesos padrões
	detector = MTCNN()
	# detecta a face na iagem
	results = detector.detect_faces(pixels)
	# extrai a caixa delimitadora da primeira imagem 
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extrai a face
	face = pixels[y1:y2, x1:x2]
	# reorganiza os pixels
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# Função que carrega todos as imagens de faces dos famosos presentes no diretório. Ressaltando que essa função utiliza a função anterior *load_faces*:
def load_dataset(directory):
	X, y = list(), list()
	# enumera diretorios, um por classe(ator)
	for subdir in os.listdir(directory):
		# caminho abssoluto
		path = directory + subdir + '/'
		# Pula os arquivos caso o diretório esteja vazio 
		# if not isdir(path):
		#	continue
		# carrega todos os arquivos do subdiretório
		faces = load_faces(path)
		# cria as labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# guarda as faces e labels em variáveis (dependentes e independentes)
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

# Função que extrai as faces apenas de um folder especifico e exibte na tela:
def test_extract_faces(folder):
    
    i = 1  
    for filename in os.listdir(os.path.join(os.getcwd(), folder)):
    	# caminho do arquivo
    	path = folder + filename
    	# Extrair a face
    	face = extract_face(path)
    	print(i, face.shape)
    	# Plota
    	pyplot.subplot(2, 7, i)
    	pyplot.axis('off')
    	pyplot.imshow(face)
    	i += 1
    
    pyplot.show()



def main ():
    
    test_extract_faces('face_recognition\\data\\val\\ben_afflek\\')
    
    # Carregando todas as faces das celebridades presentes do driver:
    # conjunto de treinamento
    X_train, y_train = load_dataset(os.path.join(os.getcwd(), 'face_recognition\\data\\train\\'))
    # conjunto de testes
    X_test, y_test = load_dataset(os.path.join(os.getcwd(), 'face_recognition\\data\\val\\'))
    
    print('Loaded: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
       
    # Carregando o modelo keras *facenet* que contém um processo de *featurização* das faces:
    
    model = load_model(os.path.join(os.getcwd(), 'face_recognition\\data\\model\\facenet_keras.h5'))
    
    # imprime a forma dos inputs e outputs 
    print(model.inputs)
    print(model.outputs)
    
    #Etapa de transformação das faces em features. No final temos 93 de treinamento e 25 imagens de validação contendo 128 features a respeito de 5 celebridades:
    # Ben Afflek, Elton John, Madonna, Mindy Kaling e Jerry Seinfeld.
    # etapa de featurização das faces usando o facenet
    
    X_train_new = list()
    for face_pixels in X_train:
    	embedding = get_embedding(model, face_pixels)
    	X_train_new.append(embedding)
    X_train_new = asarray(X_train_new)
    print(X_train_new.shape)
    # conversão de cada face no conjunto de testes 
    X_test_new = list()
    for face_pixels in X_test:
    	embedding = get_embedding(model, face_pixels)
    	X_test_new.append(embedding)
    X_test_new = asarray(X_test_new)
    print(X_test_new.shape)
    
    
    # Criando um Modelo SVM para classificar as imagens.
    # Salvando as imagens
    X_train_faces, y_train_faces, X_test_faces, y_test_faces  = X_train, y_train, X_test, y_test
    
    # carregando as imagens featurizadas:
    X_train, X_test = X_train_new, X_test_new
    
    
    # normalização dos input vectors
    in_encoder = Normalizer(norm='l2')
    X_train = in_encoder.transform(X_train)
    X_test = in_encoder.transform(X_test)
    
    # codificação das labels: 
    out_encoder = LabelEncoder()
    out_encoder.fit(y_train)
    y_train = out_encoder.transform(y_train)
    y_test = out_encoder.transform(y_test)
    
    # treinamento do modelo
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # teste do modelo em uma amostra randomica no conjunto de testes 
    selection = choice([i for i in range(X_test.shape[0])])
    random_face_pixels = X_test_faces[selection]
    random_face_emb = X_test[selection]
    random_face_class = y_test[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])
    
    # Realizando a predição de uma face
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    
    # Retornando o nome da face
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    print('Expected: %s' % random_face_name[0])
    
    # Plotando o valor e a imagem prevista junto com a probabilidade entre parênteses:
    pyplot.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()


if __name__ == "__main__":
    main()