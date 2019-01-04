"""

Breitbart              23781
New York Post          17493
NPR                    11992
CNN                    11488
Washington Post        11114
Reuters                10710
Guardian                8681
New York Times          7803
Atlantic                7179
Business Insider        6757
National Review         6203
Talking Points Memo     5214
Vox                     4947
Buzzfeed News           4854
Fox News                4354

"""

import operator
import string
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import glob
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile

from gensim import models


DATA_DIR = 'all-the-news/'



def tokenize_and_clean(row):
	sentence = str(row['title'])
	#tokens = nltk.word_tokenize(str(row['title']))
	sentence = sentence.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	filtered_sentence = [w for w in tokens if not w in stopwords.words()]
	return filtered_sentence


def load_publications(data_dir, publications):

	glob_template = data_dir + "*.csv"

	# load all files in directory into single df
	news_dfs = []
	for f in glob.glob(glob_template):
		df = pd.read_csv(f)
		news_dfs.append(df)
	news_df = pd.concat(news_dfs)

	# extract selected publications titles and content
	selected_publications_df = news_df[news_df['publication'].isin(publications)]
	selected_titles_and_content = selected_publications_df[['title', 'content']]

	# tokenize sentences and concatinate into single dataframe
	selected_titles_and_content['tokenized_titles'] = selected_titles_and_content.apply(lambda row: tokenize_and_clean(row), axis=1)
	#selected_titles_and_content['tokenized_titles'] = selected_titles_and_content.apply(lambda row: nltk.word_tokenize(str(row['title'])), axis=1)
	#selected_titles_and_content['tokenized_content'] = selected_titles_and_content.apply(lambda row: nltk.word_tokenize(str(row['content'])), axis=1)
	#frames = selected_titles_and_content[['tokenized_titles', 'tokenized_content']].tolist()
	#sentences_df = pd.concat(frames)

	# temporarily just use titles
	return selected_titles_and_content['tokenized_titles'].tolist()


def get_model_file(model_name):
	return "models/" + model_name + '.bin'

def get_wordvectors_file(model_name):
	return "models/" + model_name + '.kv'

def load_wv(model_name):
	f = get_wordvectors_file(model_name)
	wv = KeyedVectors.load(f, mmap='r')
	return wv

def load_model(model_name):
	f = get_model_file(model_name)
	model = Word2Vec.load(f)
	return model


def train_and_save_model(corupus, selected_sentences, model_name, size=250, epochs=10):

	model_file = get_model_file(model_name)
	wordvectors_file = get_wordvectors_file(model_name)

	#initiallizing the model
	model = Word2Vec(corupus, size=size, min_count=1)


	# maybe save the word indices?
	model.wv.save(wordvectors_file)
	wv = KeyedVectors.load(wordvectors_file, mmap='r')

	# train 
	model.train(selected_sentences, total_examples=1, epochs=epochs)

	# save model
	model.save(model_file)


EPOCHS = 1000

LIBERAL_MODEL_FILE = 'liberal_model'
CONSERVATIVE_MODEL_FILE = 'conservative_model'

liberal_publications = ['CNN','Washington Post', 'Vox']
conservative_publications = ['Fox News', 'Breitbart', 'New York Post']

#
# Build and train models
#


liberal_sentences = load_publications(DATA_DIR, liberal_publications)
conservative_sentences = load_publications(DATA_DIR, conservative_publications)

corupus = liberal_sentences + conservative_sentences

train_and_save_model(corupus, liberal_sentences, LIBERAL_MODEL_FILE, epochs=EPOCHS)
train_and_save_model(corupus, conservative_sentences, CONSERVATIVE_MODEL_FILE, epochs=EPOCHS)


#
# testing
#

conservative_wv = load_wv(CONSERVATIVE_MODEL_FILE)
liberal_wv = load_wv(LIBERAL_MODEL_FILE)


cw, cv = zip(*conservative_wv.vocab.items())
cm = load_model(CONSERVATIVE_MODEL_FILE)

lw, lv = zip(*liberal_wv.vocab.items())
lm = load_model(LIBERAL_MODEL_FILE)

word_similarities = {}
for c_word, l_word in zip(cw,lw):
	#score = np.dot(cm[c_word], lm[l_word])
	cosine_similarity = numpy.dot(cm[c_word], lm[l_word])/(numpy.linalg.norm(cm[c_word])* numpy.linalg.norm(lm[l_word]))

	if c_word == l_word:
		word_similarities[c_word] = score

	else:
		print("cword != lword")


word_sims_df = pd.DataFrame.from_dict(word_similarities, orient='index') #, index=['Word', 'Score']) #.from_dict(word_similarities)

word_sims_df[0] = word_sims_df[0].astype(float)
word_sims_df = word_sims_df.sort_values(by=[0], ascending=True)

print(word_sims_df)
word_sims_df.to_csv("conservative_liberal_world_similarity_scores.csv")


