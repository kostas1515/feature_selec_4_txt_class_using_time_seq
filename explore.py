import os
from FeatureSelection import FeatureSelection
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.feature_extraction.text import  TfidfVectorizer,TfidfTransformer,CountVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import math





def split_data(data,partition_by_id,category): #put this method under the open csv loop data must be in panda form
    data= data
    cat=category
    data['topic_bool']=data['topic'].map(lambda text :target_category(text,cat))
    for index,row in data.iterrows():
        if ( int(row['filename']) < partition_by_id ):
            y_train.append(row['topic_bool'])
        else:
            y_test.append(row['topic_bool'])


def target_category(text,target_cat):
    text=text[:-1]# strip last ;
    array=text.split(';')
    for x in array:
        if (x.startswith(target_cat)):
            return 1
    return 0


category_matrix=["C1","C11","C12","C13","C14","C15","C151","C1511","C152","C16","C17","C171","C172","C173","C174","C18","C181","C182","C183","C2","C21",
"C22","C23","C24","C3","C31","C311","C312","C313","C32","C33","C331","C34","C4","C41","C411","C42","C","E1","E11","E12","E121","E13",
"E131","E132","E14","E141","E142","E143","E2","E21","E211","E212","E3","E31","E311","E312","E313","E4","E41","E411","E5","E51","E511",
"E512","E513","E6","E61","E7","E71","E","G1","G15","G151","G152","G153","G154","G155","G156","G157","G158","G159","G","GCRIM","GDEF","GDIP","GDIS","GENT","GENV","GFAS","GHEA","GJOB","GMIL","GOBIT","GODD","GPOL","GPRO","GREL","GSCI","GSPO","GTOUR",
"GVIO","GVOTE","GWEA","GWELF","M1","M11","M12","M13","M131","M132","M14","M141","M142","M143","MCAT"]
# category_matrix=["C"]
train_list=[]
test_list=[]
i=0
for category in category_matrix:
    df=pd.DataFrame()
    y_train=[]
    y_test=[]
    partition_by_id=389827

    #use relative path
    for csv in os.listdir("../testspace2/csvs"):
        data = pd.read_csv("../testspace2/csvs/"+csv, encoding = 'iso-8859-1')
        split_data(data,partition_by_id,category)
    train_list.append(sum(y_train))
    test_list.append(sum(y_test))
        




