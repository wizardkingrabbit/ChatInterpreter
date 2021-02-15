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
# the following functions are from HW1, with little modification.
# 

def parts_of_speech(s):
	tokens = word_tokenize(s)
	tokens_and_tags = nltk.pos_tag(tokens)
	n = 0
	temp = set()
	for item in tokens_and_tags:
		if (not item[1] in temp):
			temp.add(item[1])
			n += 1
	print(" the total number of tokens is " + str(n))
	tags = [ item[1] for item in tokens_and_tags ]
	tag_counts = nltk.FreqDist(tags)
	sorted_tag_counts = tag_counts.most_common()
	for item in sorted_tag_counts:
		tag_percent = 100 * item[1]/n
		p = '{0:.2f}'.format(tag_percent)
		print('Tag:',item[0],'\t   Percentage of tokens = ', p )
	return(tokens_and_tags)

def create_bow_from_reviews(clips):
	text = []
	Y = []
	lengths = []
	print('\nExtracting tokens from each review.....(can be slow for a large number of reviews)......')   
	for clip in clips:
		review = Tokenizer_kit.Concatenate_str_list(clip.chats)
		stars = clip.get_label_binary()
		if (stars == -1):
			stars = 1
		if (stars == 1):
			text.append(review)
			Y.append('1')
		if (stars == 0):
			text.append(review)   
			Y.append('0')
	vectorizer = CountVectorizer(ngram_range = (1, 2), stop_words = 'english', min_df = 0.01)
	print("show vectorizer: ", vectorizer)
	X = vectorizer.fit_transform(text)
	print('Data shape: ', X.shape)
	return X, Y, vectorizer
    
def logistic_classification(X, Y, test_fraction):
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
	clip_list = []
	for filename in os.listdir("labeled_clip_data"):
		the_file = open("labeled_clip_data/" + filename, 'rb')
		the_pkl = pickle.load(the_file)
		for clip in the_pkl:
			clip_list.append(clip)
	xx, yy, vect = create_bow_from_reviews(clip_list)
	print(yy)
	classifier = logistic_classification(xx, yy, 0.2)
	most_significant_terms(classifier, vect, 3)
	return

if __name__ == "__main__":
    main()
