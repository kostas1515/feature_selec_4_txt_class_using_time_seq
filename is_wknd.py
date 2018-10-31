from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import PorterStemmer , WordNetLemmatizer
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import numpy as np
from datetime import datetime 

for csv in os.listdir("../testspace2/csvs"):
    year=int(csv[0:4])
    month=int(csv[4:6])
    day=int(csv[6:8])
    data = pd.read_csv("../testspace2/csvs/"+csv, encoding = 'iso-8859-1')
    if((datetime(year,month,day).weekday()==5)|(datetime(year,month,day).weekday()==6)): #5 and 6 represent saturday and sunday
        data['is_wkdn']=np.ones(data.shape[0])
    else:
        data['is_wkdn']=np.zeros(data.shape[0])
    data.to_csv("../testspace2/csvs/"+csv, encoding= 'iso-8859-1', index=False)