from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

# Removendo as tags htmls:
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Removendo alguns caracteres especiais como colchetes
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Função que remove os caracteres especiais:
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text


# Função que limpa o texto
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_special_characters(text)
    return text

flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Movies Sentiment Classifier", 
		  description = "Predict the sentiment of movies review")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'review': fields.String(required = True, 
				  							   description="Text containing the review of movies", 
    					  				 	   help="Text review cannot be blank")})

classifier = joblib.load('classifier_movies.joblib')
vectorizer = joblib.load('count_vectorizer.joblib')


@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		try: 
			formData = request.json
			#print(formData)
			#print('review:', formData['review'])
			#data = [val for val in formData.values()]
			data = [denoise_text(formData['review'])]
			data =  vectorizer.transform(data)          
			prediction = classifier.predict(data)
			label = { 0: "Negative", 1: "Positive"}
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "Prediction: " + label[prediction[0]] + " (" + str(np.round(np.max(classifier.predict_proba(data)),2)*100) + "%)"
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})