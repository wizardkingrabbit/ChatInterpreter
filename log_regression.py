#!/usr/bin/python
# -*- coding: utf-8 -*-

import nltk 
from nltk import word_tokenize
import simplejson as json
import sklearn
from sklearn.feature_extraction.text import * 
from sklearn.model_selection import train_test_split 

from sklearn import linear_model 
from sklearn import metrics 

import numpy as np
import matplotlib.pyplot as plt
import pickle
import Utilities
import Tokenizer_kit
import os

import warnings 
warnings.filterwarnings(action='ignore')

# nltk.download('stopwords') is needed
# the following functions are from HW1, with some modification.
    
def logistic_classification(X, Y, test_fraction):
	if (test_fraction == 0):
		classifier = linear_model.LogisticRegression(penalty = 'l2', fit_intercept = True)
		classifier.fit(X, Y)
		return(classifier)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
	print('Number of training examples: ', X_train.shape[0])
	print('Number of testing examples: ', X_test.shape[0])   
	print('Vocabulary size: ', X_train.shape[1])
	classifier = linear_model.LogisticRegression(penalty = 'l2', fit_intercept = True)
	print('\nTraining a model with', X_train.shape[0], 'examples.....')
	classifier.fit(X_train, Y_train)
	train_predictions = classifier.predict(X_train)
	train_accuracy = classifier.score(X_train, Y_train)
	print('\nTraining:')
	print(' accuracy:',format( 100*train_accuracy , '.2f'))
	print('\nTesting: ')
	test_predictions = classifier.predict(X_test)
	test_accuracy = classifier.score(X_test, Y_test)
	print(' accuracy:', format( 100*test_accuracy , '.2f') )
	class_probabilities = classifier.predict_proba(X_test)
	test_auc_score = sklearn.metrics.roc_auc_score(Y_test, class_probabilities[:,1])
	print(' AUC value:', format( 100*test_auc_score , '.2f') )
	return(classifier)

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

def main():
	run_mode = Utilities.prompt_for_str("Classifier based on all labeled files or a single file? (all/single)", {"all", "single"})
	if_stop = Utilities.prompt_for_str("Should I use defualt stop words or stop words built by my author? (default/author)", {"default", "author"})

	special_stop_word = None
	if (if_stop == "default"):
		pass
	if (if_stop == "author"):
		special_stop_word = {"1", "2", "11", "111111"}

	if (run_mode == "all"):
		test_frac = Utilities.prompt_for_float("Test fraction is ", 0, 1)
		text = []
		Y = []
		for filename in os.listdir("labeled_clip_data"):
			the_file = open("labeled_clip_data/" + filename, 'rb')
			the_pkl = pickle.load(the_file)
			for clip in the_pkl:
				text.append(Tokenizer_kit.Concatenate_str_list(clip.chats))
				if (clip.get_label_binary() == 0):
					Y.append(0)
				else:
					Y.append(1)

		concat_str = Tokenizer_kit.Concatenate_str_list(text)
		token_list = Tokenizer_kit.Simple_tokenizer(concat_str)
		the_bow = Tokenizer_kit.List_to_bow(token_list)

		if (special_stop_word == None):
			vect = CountVectorizer(ngram_range = (1, 2), stop_words = 'english', min_df = 0.01)
		else:
			vect = CountVectorizer(ngram_range = (1, 2), stop_words = special_stop_word, min_df = 0.01)
		X = vect.fit_transform(text)
		classifier = logistic_classification(X, Y, test_frac)

	if (run_mode == "single"):
		text = []
		Y = []
		train_file_path = Utilities.prompt_for_file("which labeled file you want to use?")
		the_file = open(train_file_path, 'rb')
		the_pkl = pickle.load(the_file)
		for clip in the_pkl:
			text.append(Tokenizer_kit.Concatenate_str_list(clip.chats))
			if (clip.get_label_binary() == 0):
				Y.append(0)
			else:
				Y.append(1)
		concat_str = Tokenizer_kit.Concatenate_str_list(text)
		token_list = Tokenizer_kit.Simple_tokenizer(concat_str)
		the_bow = Tokenizer_kit.List_to_bow(token_list)
		if (special_stop_word == None):
			vect = CountVectorizer(ngram_range = (1, 2), stop_words = 'english', min_df = 0.01)
		else:
			vect = CountVectorizer(ngram_range = (1, 2), stop_words = special_stop_word, min_df = 0.01)
		X = vect.fit_transform(text)
		classifier = logistic_classification(X, Y, 0)

		while (input("enter y to test the classifier against another labeled file, enter other to quit") == "y"):
			# this feature is not finished!
			# how can I use a classifier built according to one word set on another word set
			text = []
			Y = []
			test_file_path = Utilities.prompt_for_file("which labeled file you want to use?")
			the_file = open(test_file_path, 'rb')
			the_pkl = pickle.load(the_file)
			for clip in the_pkl:
				text.append(Tokenizer_kit.Concatenate_str_list(clip.chats))
				if (clip.get_label_binary() == 0):
					Y.append(0)
				else:
					Y.append(1)
			concat_str = Tokenizer_kit.Concatenate_str_list(text)
			token_list = Tokenizer_kit.Simple_tokenizer(concat_str)
			the_bow = Tokenizer_kit.List_to_bow(token_list)
			if (special_stop_word == None):
				vect = CountVectorizer(ngram_range = (1, 2), stop_words = 'english', min_df = 0.01)
			else:
				vect = CountVectorizer(ngram_range = (1, 2), stop_words = special_stop_word, min_df = 0.01)
			X = vect.fit_transform(text)
			test_accuracy = classifier.score(X, Y)
			print("the test accuracy is " + str(test_accuracy))

	if (input("enter y to look at top 5 significant terms, enter other to quit") == "y"):
		most_significant_terms(classifier, vect, 5)
	return

if __name__ == "__main__":
    main()