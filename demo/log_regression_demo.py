#!/usr/bin/python
# -*- coding: utf-8 -*-

# modules used
import nltk 
from nltk import word_tokenize
import simplejson as json

import sklearn
from sklearn.feature_extraction.text import * 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model 
from sklearn import metrics 
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from Utilities import *
from Tokenizer_kit import *
from Embedding import *
from random import shuffle
from nltk.corpus import stopwords

# used by word purification process
# nltk.download('stopwords') is needed
import enchant
global_dict = enchant.Dict("en_US")
global_slang = set({"F", "???", "!!!", "!?", "pog", "nice", "noice", "haha", "lol", "lul", "lmao", "yes", "noo", "no", "yeah", "ree", "oof", "pogu", "xd", "ez", "money", "GG", "gg"})
nltk_stop_words = set(stopwords.words('english'))

import warnings 
warnings.filterwarnings(action='ignore')

# the following 2 functions are from HW1, with some modification.
def logistic_classification(X, Y, classifier = None):
	msg_line = ""
	if (classifier == None):
		mode = "Training"
		msg_line += f"Number of training examples: [{X.shape[0]}]" + os.linesep
		msg_line += f"Vocabulary size: [{X.shape[1]}]" + os.linesep
		classifier = linear_model.LogisticRegression(penalty = 'l2', tol = 0.3, solver = "sag", max_iter = 10)
		classifier.fit(X, Y)
	else:
		mode = "Validation/Testing"
	accuracy = classifier.score(X, Y)
	msg_line += mode + f" accuracy: [{format( 100*accuracy , '.2f')}]" + os.linesep
	train_predictions = classifier.predict(X)
	class_probabilities = classifier.predict_proba(X)
	test_auc_score = sklearn.metrics.roc_auc_score(Y, class_probabilities[:,1])
	msg_line += mode + f" AUC value: [{format( 100*test_auc_score , '.2f')}]" + os.linesep
	default_counter = 0
	for i in Y:
		if (i == 0):
			default_counter += 1
	default_accuracy = default_counter / len(Y)
	msg_line += f" default accuracy: [{format( 100*default_accuracy , '.2f')}]" + os.linesep
	counter = 0
	my_error = []
	while (counter < X.shape[0]):
		if (train_predictions[counter] != Y[counter]):
			my_error.append(counter)
		counter += 1
	return classifier, my_error, msg_line

def most_significant_terms(classifier, vectorizer, K):
	count = 0
	topK_pos_weights = set()
	topK_pos_terms = set()
	while(count < K):
		max = -1
		temp_count = 0
		temp_term = "null indicator, if the proper word is not found"
		for weight in classifier.coef_[0]:
			if (weight > 0 and weight > max and not weight in topK_pos_weights):
				max = weight
				temp_term = vectorizer.get_feature_names()[temp_count]
			temp_count += 1
		if (not max == -1):
			topK_pos_weights.add(max)
			topK_pos_terms.add(temp_term)
			print("Positive weight rank ", str(count + 1), ": ")
			print("--->", temp_term, ", and its weight is: ", str(max))
		count += 1
	count = 0
	topK_neg_weights = set()
	topK_neg_terms = set()
	while(count < K):
		min = 1
		temp_count = 0
		temp_term = "null indicator, if the proper word is not found"
		for weight in classifier.coef_[0]:
			if (weight < 0 and weight < min and not weight in topK_neg_weights):
				min = weight
				temp_term = vectorizer.get_feature_names()[temp_count]
			temp_count += 1
		if (not min == 1):
			topK_neg_weights.add(min)
			topK_neg_terms.add(temp_term)
			print("Negative weight rank ", str(count + 1), ": ")
			print("--->", temp_term, ", and its weight is: ", str(min))
		count += 1
	return(topK_pos_weights, topK_neg_weights, topK_pos_terms, topK_neg_terms)

# used by demo
def demo_convert(be_converted, mode):
	word_list = be_converted.split(" ")
	result = ""
	if (mode == "1" or mode == 1):
		for word in word_list:
			result += my_translator_simple(word)
			result += " "
	elif (mode == "2" or mode == 2):
		for word in word_list:
			result += my_translator(word)
			result += " "
	return result

# train log regression model with the data provided
def train_log_regression(X, Y):
	classifier = linear_model.LogisticRegression(penalty = 'l2', tol = 0.3, solver = "sag", max_iter = 10)
	classifier.fit(X, Y)
	return classifier

# get validation error rate and default error rate of a model with the data provided
def validate(classifier, X, Y):
	accuracy = classifier.score(X, Y)
	default_counter = 0
	for i in Y:
		if (i == 0):
			default_counter += 1
	default_accuracy = default_counter / len(Y)
	return accuracy, default_accuracy

# directly convert a list of long strings into a one-hot vector
# it does both tokenization and vectorization
# it should returns an 2-D array
# X index the clip, Y index the token
# However, since countVectorizer gives a better result (it does a better job tokenizing stuff)
# this function is not used
def to_ohv(text_list, stop_words = [], min_len = 2):
	token_set = set()
	for text in text_list:
		for word in text.split():
			if (len(word) > min_len) and (not word in stop_words) and (not word in set(stopwords.words('english'))) and (not word in token_set):
				token_set.add(word)
	result = np.zeros((len(text_list), len(token_set)))
	for i, token in enumerate(token_set):
		for ii, sentence in enumerate(text_list):
			if (token in sentence):
				result[ii][i] = 1
	return result

# translate a word into something standard
def my_translator(target_word, stop_words = nltk_stop_words):
	result = target_word
	if global_dict.check(target_word):
		# this word is a standard word, return it
		result =  target_word
	elif (target_word in global_slang):
		# this word is not a standard word, is it an internet slang?
		result =  target_word
	elif (Embedding_word_modifier(target_word) in global_slang):
		# Or it could be some special form of an iternet slang
		result =  Embedding_word_modifier(target_word)
	elif len(global_dict.suggest(target_word)) > 0:
		# it is nothing but there are similar words
		result =  global_dict.suggest(target_word)[0]
	else:
		# it is nothing, probably an emote
		# but we do not have a similar word to it, so return itself
		result = target_word
	try:
		# is it a number? Maybe we should purify numbers
		_test = float(target_word)
		result = "NUMBER_WORD"
	except:
		pass
	if result == None:
		result =  target_word
	else:
		result =  result
	# last step : remove redundant consequtive words
	real_result = []
	last_letter = None
	for letter in result:
		if (not last_letter == None) and (letter == last_letter):
			pass
		else:
			real_result.append(letter)
		last_letter = letter
	return Concatenate_str_list(real_result, splitter = '')

# simple version of my translator. there are fewer steps.
def my_translator_simple(target_word, stop_words = nltk_stop_words):
	result = target_word
	if global_dict.check(target_word):
		# this word is a standard word, return it
		result =  target_word
	elif (target_word in global_slang):
		# this word is not a standard word, is it an internet slang?
		result =  target_word
	elif (Embedding_word_modifier(target_word) in global_slang):
		# Or it could be some special form of an iternet slang
		result =  Embedding_word_modifier(target_word)
	elif len(global_dict.suggest(target_word)) > 0:
		# it is nothing but there are similar words
		result =  global_dict.suggest(target_word)[0]
	else:
		# it is nothing, probably an emote
		# but we do not have a similar word to it, so return itself
		result = target_word
	return result
        
# utility function, you give a path to it and it add all .pkl file under that path to original set
# please be sure that you give it a valid path when using it
def add_filepath_to_set(the_path:str, is_file:bool, original_set):
    if (is_file):
        original_set.append(the_path)
    else:
        for entry in os.scandir(the_path):
            if (entry.path.endswith(".pkl") and entry.is_file()):
                original_set.append(entry.path)
    return original_set

# interpret a pkl file and extract its data into three lists
def add_clipdata_to_set(clip_list, text_list, y_list, pkl_path, do_convert = 0, filter_stopword = True, show_debug = False):
	the_file = open(pkl_path, 'rb')
	the_pkl = pickle.load(the_file)
	for clip in the_pkl:
		clip_list.append(clip)
		if do_convert == 0:
			# no conversion, totally original text
			text_list.append(Concatenate_str_list(clip.chats))
		elif do_convert == 1:
			# massive translation
			temp_text = []
			for chat in clip.chats:
				for word in chat.split():
					temp_word = my_translator(word)
					temp_text.append(temp_word)
			if (show_debug):
				print(temp_text)
			text_list.append(Concatenate_str_list(temp_text))
		elif do_convert == 2:
			# minimal translation, faster
			temp_text = []
			for chat in clip.chats:
				for word in chat.split():
					temp_word = my_translator_simple(word)
					temp_text.append(temp_word)
			if (show_debug):
				print(temp_text)
			text_list.append(Concatenate_str_list(temp_text))
		if (clip.get_label_binary() == 0):
			y_list.append(0)
		else:
			y_list.append(1)
	the_file.close()
	return clip_list, text_list, y_list

# randomize data
def randomize_data(clip_list, text_list, y_list):
	order_list = list(range(len(clip_list)))
	random.shuffle(order_list)
	new_clip = list(clip_list)
	new_text = list(text_list)
	new_y = list(y_list)
	for i, ii in enumerate(order_list):
		new_clip[i] = clip_list[ii]
		new_text[i] = text_list[ii]
		new_y[i] = y_list[ii]
	return new_clip, new_text, new_y

# this function iteratively run the main to find the best param
def best_param(ngram, panelty, dual, tol, C, fit_intercept, solver, max_iter, num_iter = 10, test_ratio = 0.2, test_on = ["labeled_clip_data/Teo"]):
	va_err_list = []
	#define training set
	filepath = []
	text = []
	Y = []
	all_clip = []
	for path in test_on:
		filepath = add_filepath_to_set(path, False, filepath)
	for filename in filepath:
		all_clip, text, Y = add_clipdata_to_set(all_clip, text, Y, filename)
	#define validation set
	training_size = int(len(Y) * (1 - test_ratio))
	validation_size = len(Y) - training_size
	# iteratively test the model
	while (num_iter > 0):
		# randomize the data
		all_clip, text, Y = randomize_data(all_clip, text, Y)
		# construct the vectorizer
		vect = CountVectorizer(ngram_range = (1, ngram), stop_words = 'english', min_df = 0.01, tokenizer = Embedding_tokenize)
		X = vect.fit_transform(text)
		# make classifier
		# the following line is responsible for taking different parameters
		classifier = linear_model.LogisticRegression(C = C, dual = dual, penalty = panelty, fit_intercept = fit_intercept, tol = tol, solver = solver, max_iter = max_iter)
		classifier.fit(X[:training_size], Y[:training_size])
		va_err_list.append(classifier.score(X[training_size:], Y[training_size:]))
		num_iter -= 1
	return np.average(va_err_list)

# main
# you do not need to run sudo main if you already have proper input data
def main():
	text = []
	Y = []
	all_clip = []
	# define data set
	filepath = []
	file_or_folder, _type = prompt_for_file_folder("enter a path to a file or a folder to add that to the training set, enter e to exit", {"e"})
	while(file_or_folder != "e"):
		filepath = add_filepath_to_set(file_or_folder, _type == "file", filepath)
		file_or_folder, _type = prompt_for_file_folder("enter a path to a file or a folder to add that to the training set, enter e to exit", {"e"})
	for filename in filepath:
		all_clip, text, Y = add_clipdata_to_set(all_clip, text, Y, filename)
	#define validation set
	validation_ratio = prompt_for_float("What proportion of the training data would be used for validation?", 0, 1)
	training_size = int(len(Y) * (1 - validation_ratio))
	validation_size = len(Y) - training_size
	# randomize the data
	all_clip, text, Y = randomize_data(all_clip, text, Y)
	# define stop word
	special_stop_word = set(stopwords.words('english'))
	# construct the vectorizer
	if_tfidf = prompt_for_str("Do you want to use tfidf on ohv construction? (yes/no)", {"yes","no"})
	if (if_tfidf == "yes"):
		vect = TfidfVectorizer(ngram_range = (1, 2), stop_words = special_stop_word, min_df = 0.01, tokenizer = Embedding_tokenize)
	if (if_tfidf == "no"):
		vect = CountVectorizer(ngram_range = (1, 2), stop_words = special_stop_word, min_df = 0.01, tokenizer = Embedding_tokenize)
	X = vect.fit_transform(text)
	# make classifier
	classifier = train_log_regression(X[:training_size], Y[:training_size])
	accu, default_accu = validate(classifier, X[training_size:], Y[training_size:])
	# look at result
	if (input("enter y to look at top 5 significant terms, enter other to quit") == "y"):
		most_significant_terms(classifier, vect, 5)
	print ("validation accuracy rate is -> " + str(accu))
	print ("default validation accuracy rate is -> " + str(default_accu))
	# test the classifier
	training_size = len(Y)
	while (input("Do you want to test this classifier on any unlabled clip data? (y/n)") == "y"):
		training_size = len(Y)
		file_path = prompt_for_file("which file you want to do test on? ")
		all_clip, text, Y = add_clipdata_to_set(all_clip, text, Y, file_path)
		if (if_tfidf == "yes"):
			vect = TfidfVectorizer(ngram_range = (1, 2), stop_words = special_stop_word, min_df = 0.01, tokenizer = Embedding_tokenize)
		if (if_tfidf == "no"):
			vect = CountVectorizer(ngram_range = (1, 2), stop_words = special_stop_word, min_df = 0.01, tokenizer = Embedding_tokenize)
		X = vect.fit_transform(text)
		classifier = train_log_regression(X[:training_size], Y[:training_size])
		predictions = classifier.predict(X[training_size:])
		counter = 0
		while(counter < len(predictions)):
			all_clip[counter].labeled = predictions[counter]
			counter += 1
		file_path = prompt_for_save_file(dir_path='model_labeled_result', f_format='.pkl')
		with open(file_path, 'wb') as f: 
			pickle.dump(all_clip, f)

if __name__ == "__main__":
    main()