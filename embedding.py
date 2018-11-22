#!/usr/bin/env python3

__author__  = "LJ Brown"
__file__ = "embedding.py"

# internal
import math
import json
import os

# dir
from headlines import article_headlines

# external
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD
#from keras.preprocessing.text import one_hot
#from keras.layers import Dense, Dropout, Convolution1D, Flatten, Input, Activation, Embedding,MaxPooling1D
#from keras.optimizers import SGD

OUTPUT = True
TEST_2_SET_RATIO = 0.2
EPOCHS = 1000
model_file = "../data/model_files/m1.json"
model_weights_file = "../data/model_files/m1.h5"

#
# helper methods
#

def score_model(x, y, model, name, batch_size):
	score = model.evaluate(x_train, y_train,
					   batch_size=batch_size, verbose=1)

	for i,metric in enumerate(model.metrics_names):
		print("%s %s: %.2f%%" % (name, metric, score[i]))


# select sources
selected_sources = ['Fox News', 'MSNBC', 'Reuters']

# load into pandas dataframe
selected_news_df = article_headlines(selected_sources)

#
# encode X and y
#

# y is a one hot encoded representation of the classes (sources)
encoder = LabelBinarizer()
y = encoder.fit_transform(selected_news_df['source'])

# X contains encoded docs (bag of words) where
# each row is a document
# each column represents a unique word in the vocabulary

# create the tokenizer
t = Tokenizer()

# fit the tokenizer on the documents
t.fit_on_texts(selected_news_df['title'])
vocab_size = len(t.word_index)

# integer encode documents (bag of words)
# each row is a document
# each column represents a unique word in the vocabulary
X = t.texts_to_matrix(selected_news_df['title'], mode='count')

#
# create training and test data
#
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_2_SET_RATIO)

if OUTPUT:
	print(len(x_train), 'training instances')
	print(len(x_test), 'testing intances')
	print("split (train/total) = ", x_test.shape[0]/X.shape[0])

# set dynmically
batch_size = int(np.sqrt(x_train.shape[0]))
embedding_dims = math.floor(math.sqrt(vocab_size))

# add batch dim

#
# Construct network
#

model = Sequential()

# add embedding layer
model.add(Embedding(vocab_size, embedding_dims, input_length=(vocab_size+1)))
model.add(Flatten())

model.add(Dense(20))
model.add(Activation('relu'))

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))

# output layer
model.add(Dense(len(encoder.classes_)))
model.add(Dropout(0.3))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

if OUTPUT:
  print(model.summary())

# display initial scores
score_model(x_train, y_train, model, "INITIAL train", batch_size)
score_model(x_test, y_test, model, "INTITIAL test", batch_size)


# train
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_size, verbose=2)

score_model(x_train, y_train, model, "FINAL train", batch_size)
score_model(x_test, y_test, model, "FINAL test", batch_size)

#
# plot training history
#

from matplotlib import pyplot

print(history.history.keys())

pyplot.plot(history.history['acc'], label='accuracy')
pyplot.plot(history.history['loss'], label='categorical_crossentropy')

pyplot.legend()
pyplot.show()

#
# save model
#

from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open(model_file, "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(model_weights_file)
 
#
# load model
#
 
# load json and create model
json_file = open(model_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(model_weights_file)

# evaluate loaded model on test data
#loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae']) 
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

score_model(x_train, y_train, loaded_model, "Loaded train", batch_size)
score_model(x_test, y_test, loaded_model, "Loaded test", batch_size)
