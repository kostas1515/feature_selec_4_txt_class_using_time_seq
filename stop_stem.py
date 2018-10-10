from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import PorterStemmer , WordNetLemmatizer
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import numpy as np


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def remove_sw(word):
	words=word.lower()
	words=' '.join(re.sub("([@#][A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(\d+)"," ",words).split())
	words = word_tokenize(words)
	cl_txt = ''
	for w in words:
		#w = lemmatizer.lemmatize(w,pos = "a")       #lemmatize words --  
		if w not in stop_words:
			w =stemmer.stem(w)
			cl_txt = cl_txt +' ' + ''.join(w)       #CREATE NEW CLEAN READY TXT
	return cl_txt





for csv in os.listdir("C:/Users/Kostas/Desktop/testspace/csvs"):
	data = pd.read_csv("C:/Users/Kostas/Desktop/testspace/csvs/"+csv, encoding = 'iso-8859-1')
	data = data.replace(np.nan, '', regex=True)#this replace the nan with '' in 51860 19960913
	data['title']=data['title'].map(lambda text :remove_sw(text))
	data['text']=data['text'].map(lambda text :remove_sw(text))
	data.to_csv("C:/Users/Kostas/Desktop/testspace1/"+csv, encoding= 'iso-8859-1', index=False)