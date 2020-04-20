# -*- coding: utf-8 -*-
"""
Created on Tue May 9 14:07:38 2017
For: UMR TETIS RESEARCH UNIT
Author: Gaurav_Shrivastava
"""
import os
import sys
import numpy as np
import pandas as pd
import nltk
import re
import matplotlib as plt
import json
from collections import defaultdict
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import argparse
'''for generating the proper tag for lemmatizer'''

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None

'''converts the text corpora into list of tokens'''
def pre_processing(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    #sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

'''Part of speech tagging for documents to preserve context for lemmatizer'''
def pos_tag(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

'''Utility function to flatten list of lists into single list'''
def flatten(lists):
    newlist = []
    for item in lists:
        for index in range(len(item)):
            newlist.append(item[index])
    return newlist

'''reading text corpus'''
def reading_valorcarn_corpus(filename):
	f = open(filename)
	string = f.read()
	docs = string.split("##########END##########")
	return docs

'''reading list of terms'''
def read_file(filename):
	f = open(filename)
	string = f.read()
	word = string.split('\n')
	return word


def normalise(words,tags):
	"""Normalises words to lowercase, stems and lemmatizes it.(input is a word)"""
	normalised = defaultdict(list)
	counter = 0
	for i in range(len(words)):
		word = words[i].lower()
		#
		if penn_to_wn(tags[i][1]) != None:
			word = lemmatizer.lemmatize(word,pos = penn_to_wn(tags[i][1]))
		word = stemmer.stem(word)
		normalised[word].append(counter)
		counter = counter + 1
	return normalised


def list_normalize(words):
	"""Normalises words to lowercase,
	 stems and lemmatizes it.(input is list of words)"""
	normalised = defaultdict(list)
	counter = 0
	for i in range(len(words)):
		word = words[i].lower()
		#if penn_to_wn(tags[i][1]) != None:
		word = lemmatizer.lemmatize(word)#,pos = penn_to_wn(tags[i][1]))
		word = stemmer.stem(word)
		normalised[word].append(counter)
		counter = counter + 1
	return normalised


def normalize(word):
	word = lemmatizer.lemmatize(word)#,pos = penn_to_wn(tags[i][1]))
	word = stemmer.stem(word)
	return word


def extract_singles_variation(words,norm_dict,filtered_words):
	singles = defaultdict(list)
	for word in words:
		if ' ' not in word:
			temporary_extract = extract_singles(word,norm_dict,filtered_words)
			try:
				singles[word].append(list(set(temporary_extract)))
			except:
				print("Error on extract singles variation with word: "+word)
	return singles


def extract_singles(word, norm_dict, filtered_words):
	word = normalize(word)
	if word in norm_dict:
		temp = norm_dict[word]
		word_list = []
		for entry in temp:
			word_list.append(filtered_words[entry])
		return word_list
	return None


def extract_couples_variation(words, norm_dict,filtered_words,k):
	couples = defaultdict(list)
	for word in words:
		if ' ' in word:
			temp = word.split(' ')
			if len(temp) == 2:
				word1 = temp[0]
				word2 = temp[1]
				temporary_extract = extract_couples(word1,word2, norm_dict,filtered_words,k)
				print(word,temporary_extract)
				if temporary_extract != None:
					couples[word].append(temporary_extract)
	return couples


def extract_couples(word1, word2, norm_dict, filtered_words,k):
	#find root for both words
	word1 = normalize(word1)
	word2 = normalize(word2)
	if word1 in norm_dict:
		if word2 in norm_dict:
			word_set = set([])
			#extract the occurances of the root word in the corpus
			instance1 = norm_dict[word1]
			instance2 = norm_dict[word2]
			#matching the word with at most k words occuring between the root of the two words
			extracted = matching(instance1,instance2,k)
			for extract in extracted:
				terms = terms_extract(extract,filtered_words)
				word_set.add(terms)
			return list(word_set)
	return None


def terms_extract(extract,filtered_words):
	terminology =''
	for entry in range(extract[0],extract[1]):
		terminology = terminology + filtered_words[entry] + ' '
	terminology = terminology + filtered_words[extract[1]]
	return terminology


def matching(array1, array2,k):
	extracted = []
	for entry in array1:
		for i in range(entry - k, entry + k):
			if i in array2:
				mini = min([entry,i])
				maxi = max([entry,i])
				extracted.append([mini,maxi])
				break
	return extracted


def reading_corpus(filename):
	f= open(filename)
	string = f.read()
	return string

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="give input text corpus")
	parser.add_argument("list", help="input terminology list")
	parser.add_argument("-k", "--K_value", help="give K_value(int), Default is 3", default=3)
	parser.add_argument("-s", "--size", help="slicing the size of input terminology list to: default = 100", default=100)
	args = parser.parse_args()
	filename = args.input
	if not os.path.isfile(filename):
	        print('Give correct path to text corpora ""\(-_-)/""')
	        return None
	#filename = 'Valorcarn_web_corpus.txt'
	docs = reading_corpus(filename)#reading_valorcarn_corpus(filename)
	words = []
	tags = []
	'''
	for doc in docs:
		words.append(pre_processing(doc))
		tags.append(pos_tag(doc))
	'''
	words = pre_processing(docs)
	tags = pos_tag(docs)

	words = flatten(words)
	tags = flatten(tags)
	#words = flatten(words)
	#tags = flatten(tags)
	global stemmer, lemmatizer
	stemmer = nltk.stem.porter.PorterStemmer()
	lemmatizer = nltk.WordNetLemmatizer()

	filtered_words = [word for word in words if word not in stopwords.words('english')]
	filtered_tags = [word for word in tags if word[0] not in stopwords.words('english')]

	normalised = normalise(filtered_words,filtered_tags)
	list_name = args.list
	if not os.path.isfile(filename):
	        print('Give correct path to input list ""\(-_-)/"" check --help')
	        return None
	lists = read_file(list_name)
	singles = extract_singles_variation(lists,normalised,filtered_words)
	#couples = extract_couples_variation(lists,normalised,filtered_words)
	normalised = normalise(words,tags)
	size = int(args.size)
	K_value = int(args.K_value)
	couples = extract_couples_variation(lists[:size],normalised,words,K_value)
	keys = set(couples.keys())
	for key in keys:
	    if len(couples[key][0]) ==1:
	        del couples[key]
	open('driven_extraction_version_1.json','w').write(json.dumps(couples,indent = 4))

if __name__ == '__main__':
	main()