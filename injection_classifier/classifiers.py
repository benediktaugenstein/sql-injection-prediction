class InjectionPredictor:
	def __init__(self):
		import pickle
		import os

		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

		dirname = os.path.dirname(__file__)
		self.model_path = os.path.join(dirname, 'models/injection_classifier/injection_classifier_model.h5')
		self.load_model_path = os.path.join(dirname, 'models/injection_classifier/lm.pickle')
		self.tokenizer_path = os.path.join(dirname, 'models/injection_classifier/injection_classifier_tokenizer.pickle')
		self.ps_path = os.path.join(dirname, 'models/injection_classifier/injection_classifier_ps.pickle')
		self.ohe_path = os.path.join(dirname, 'models/injection_classifier/injection_classifier_ohe.pickle')

		with open(self.load_model_path, 'rb') as handle:
			self.load_model = pickle.load(handle)
		with open(self.tokenizer_path, 'rb') as handle:
			self.tokenizer = pickle.load(handle)
		with open(self.ps_path, 'rb') as handle:
			self.ps = pickle.load(handle)
		with open(self.ohe_path, 'rb') as handle:
			self.ohe = pickle.load(handle)

		#self.model = self.load_model("injection_classifier_model.h5")
		import tensorflow as tf
		self.model = tf.keras.models.load_model(self.model_path)

	def predict(self, user_inp_initial):

		user_inp = list(user_inp_initial)
		user_inp = [user_inp]
		user_inp = self.tokenizer.texts_to_sequences(user_inp)
		user_inp = self.ps(user_inp, maxlen=238, padding='post', truncating='post')

		prediction = self.model.predict(user_inp)
		#prediction_inverse_transformed = self.ohe.inverse_transform(prediction)
		#print(prediction_inverse_transformed)
		#var = prediction_inverse_transformed[0]

		if prediction[0][0] < 0.5:
			confidence = ((0.5 - prediction[0][0]) / 0.5) * 50 + 50
			prediction = 'SQL Injection'
			confidence = round(confidence, 2)
		elif prediction[0][0] > 0.5:
			confidence = ((prediction[0][0] - 0.5) / 0.5) * 50 + 50
			prediction = 'username/password'
			confidence = round(confidence, 2)

		return {'prediction':prediction, 'confidence':confidence}
