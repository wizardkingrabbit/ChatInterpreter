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
import template_learner

import warnings 
warnings.filterwarnings(action='ignore')

# nltk.download('stopwords') is needed
# the following functions are from HW1, with some modification.
def logistic_classification(X, Y, classifier = None):
	msg_line = ""
	if (classifier == None):
		mode = "Training"
		msg_line += f"Number of training examples: [{X.shape[0]}]" + os.linesep
		msg_line += f"Vocabulary size: [{X.shape[1]}]" + os.linesep
		classifier = linear_model.LogisticRegression(penalty = 'l2', fit_intercept = True)
		classifier.fit(X, Y)
	else:
		mode = "Validation/Testing"
	accuracy = classifier.score(X, Y)
	msg_line += mode + f" accuracy: [{format( 100*accuracy , '.2f')}]" + os.linesep
	train_predictions = classifier.predict(X)
	class_probabilities = classifier.predict_proba(X)
	test_auc_score = sklearn.metrics.roc_auc_score(Y, class_probabilities[:,1])
	msg_line += mode + f" AUC value: [{format( 100*test_auc_score , '.2f')}]" + os.linesep
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

def main(the_text = None, the_y = None, t_size = None, v_size = None):
	# use the template learner for first few steps
	if (the_text == None):
		template_learner.main_alter("linear")
	# define stop word
	if_stop = Utilities.prompt_for_str("Do you want to use default english stopwords or stopwords given by my author? (default/author)", {"default","author"})
	special_stop_word = None
	if (if_stop == "default"):
		pass
	if (if_stop == "author"):
		special_stop_word = {"1", "2", "11", "111111", "gg", "gg gg", "LUL", "LOL"}
	#concat_str = Tokenizer_kit.Concatenate_str_list(text)
	#token_list = Tokenizer_kit.Simple_tokenizer(concat_str)
	#the_bow = Tokenizer_kit.List_to_bow(token_list)
	# construct the vectorizer
	if (special_stop_word == None):
		vect = CountVectorizer(ngram_range = (1, 2), stop_words = 'english', min_df = 0.01)
	else:
		vect = CountVectorizer(ngram_range = (1, 2), stop_words = special_stop_word, min_df = 0.01)
	X = vect.fit_transform(the_text)
	classifier, t_err, t_msg = logistic_classification(X[:t_size], the_y[:t_size])
	_c, v_err, v_msg = logistic_classification(X[t_size:], the_y[t_size:], classifier)
	# look at result
	if (input("enter y to look at top 5 significant terms, enter other to quit") == "y"):
		most_significant_terms(classifier, vect, 5)
	return classifier, t_err, v_err, t_msg, v_msg

if __name__ == "__main__":
    main()