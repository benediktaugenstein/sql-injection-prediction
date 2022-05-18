import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.ExcelFile('SQL_Injection_Classifier.xlsx')
df = data.parse("data_new")
sentences = df['Input']
labels = df['Output']

for i, inp in enumerate(sentences):
    sentences[i] = list(str(inp))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

sequences = pad_sequences(sequences, padding='post', truncating='post')
word_count = len(word_index)

len_input = len(sequences[100])

ohe = LabelBinarizer()
labels_transformed = ohe.fit_transform(labels)
len_output = len(labels_transformed[0])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=word_count+1, output_dim=16, input_length=len_input),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

train_sequences = np.append(sequences[:850], sequences[1150:], axis=0)
train_labels_transformed = np.append(labels_transformed[:850], labels_transformed[1150:], axis=0)

num_epochs = 100
history = model.fit(train_sequences, train_labels_transformed, epochs=num_epochs, verbose=1, batch_size=150, validation_data=(sequences[850:1150], labels_transformed[850:1150]), shuffle=True)

def new_input():
    user_inp_initial = input("Input here: ")
    user_inp_initial = list(user_inp_initial)
    user_inp = [user_inp_initial]
    user_inp = tokenizer.texts_to_sequences(user_inp)
    user_inp = pad_sequences(user_inp, maxlen=len_input, padding='post', truncating='post')

    prediction = model.predict(user_inp)

    prediction_inverse_transformed = ohe.inverse_transform(prediction)
    var = prediction_inverse_transformed[0]

    if prediction[0][0]<0.5:
        confidence = ((0.5-prediction[0][0])/0.5) * 100
        print('Prediction: SQL-Injection')
        print('Confidence: ', "%.2f" % confidence, '%\n')
        new_input()
    elif prediction[0][0]>0.5:
        confidence = ((prediction[0][0]-0.5) / 0.5) * 100
        print('Prediction: username/password')
        print('Confidence:', "%.2f" % confidence, '%\n')
        new_input()

new_input()
