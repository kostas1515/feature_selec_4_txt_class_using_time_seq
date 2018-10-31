import os
from FeatureSelection import FeatureSelection
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.feature_extraction.text import  TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import math


category_matrix=["C","C1","C2","C3","C4","E","E1","E2","E3","E4","E5","E6","G","G1","GCRIM","GDIS","GMIL","GFAS","GPRO","GSCI","GSPO","GVIO","GVOTE","GWEA","M","M11","M12","M13","M14"]
# category_matrix=["C"]
for category in category_matrix:
    df=pd.DataFrame()

    bench=FeatureSelection(category,26150) #enter target category and the last id of the preffered train_set
    #use relative path
    for csv in os.listdir("../testspace2/csvs"):
        data = pd.read_csv("../testspace2/csvs/"+csv, encoding = 'iso-8859-1')
        bench.split_data(data)


    label_train=bench.y_train
    label_test=bench.y_test


    new_x_train,new_x_test=bench.rdf(None)
    limit=len(bench.rdf_rel_pool)
    k=limit
    acc_r=[]
    axes=[]
    percent=80
    rel_pool=bench.rdf_rel_pool
    while(k>limit*0.001):
        if(k==limit):
            vectorizer = TfidfVectorizer(lowercase=False)
            n_x = vectorizer.fit_transform(new_x_train)
            clf = svm.LinearSVC().fit(n_x, label_train)
            array3=vectorizer.transform(new_x_test)
            test_test_predict = clf.predict(array3)
            acc_r.append(accuracy_score(label_test, test_test_predict))
            k=limit*80/100
            upperlimmit=None
            axes.append(100)
        else:
            pool_2_sub=rel_pool[math.ceil(k):upperlimmit]
            upperlimmit=math.ceil(k)-1
            
            #feature removal, new set extraction
            n_x_train,new_x_test=bench.transform_features(pool_2_sub,new_x_train,new_x_test)
            
            #classification
            vectorizer = TfidfVectorizer(lowercase=False)
            n_x = vectorizer.fit_transform(n_x_train)
            clf = svm.LinearSVC().fit(n_x, label_train)
            array3=vectorizer.transform(new_x_test)
            test_test_predict = clf.predict(array3)


            acc_r.append(accuracy_score(label_test, test_test_predict))
            axes.append(percent)
            percent=percent/2
            k=k/2

    df["axes"]=axes
    df["rdf_acc"]=acc_r



    



    ################## SINGLE UNIFORM ##############################

    new_x_train,new_x_test=bench.uniform('single',decision_thres=0.5,topk=None)
    # bench.rdf(topk=1000)
    # bench.uniform('single',decision_thres=0.5,topk=1000)
    # bench.random_select(1000)
    # new_x_train=bench.x_train #for chi squere only

    limit=len(bench.uniform_feat_pool)
    k=limit
    rel_pool=bench.uniform_feat_pool
    acc=[]
    while(k>limit*0.001):
        if(k==limit):
            vectorizer = TfidfVectorizer(lowercase=False)
            n_x = vectorizer.fit_transform(new_x_train)
            clf = svm.LinearSVC().fit(n_x, label_train)
            array3=vectorizer.transform(new_x_test)
            test_test_predict = clf.predict(array3)
            acc.append(accuracy_score(label_test, test_test_predict))
            k=limit*80/100
            upperlimmit=None
        else:
            pool_2_sub=rel_pool[math.ceil(k):upperlimmit]
            upperlimmit=math.ceil(k)-1
            
            #feature removal, new set extraction
            new_x_train,new_x_test=bench.transform_features(pool_2_sub,new_x_train,new_x_test)

            vectorizer = TfidfVectorizer(lowercase=False)
            n_x = vectorizer.fit_transform(new_x_train)
            clf = svm.LinearSVC().fit(n_x, label_train)
            array3=vectorizer.transform(new_x_test)
            test_test_predict = clf.predict(array3)
            acc.append(accuracy_score(label_test, test_test_predict))
            k=k/2

    df["uniform_acc"]=acc


    ##################  CHI2 ##############################

    new_x_train=bench.x_train
    x_test=bench.x_test

    k=100.2#ignore that
    limit=1000
    acc_x=[]
    while(k>0.001*limit):
        if(k==100.2):
            vectorizer = TfidfVectorizer(lowercase=False)
            n_x = vectorizer.fit_transform(new_x_train)
            limit=len(vectorizer.get_feature_names())
            features=vectorizer.get_feature_names()
            ch2 = SelectKBest(chi2, k="all")
            n_x = ch2.fit_transform(n_x, label_train)
            scores=ch2.scores_ #get the list of scores and corresponding features

            d = {'feat': features,'score': scores}# make a list of best features regarding chi2   
            feat_score = pd.DataFrame(data=d)
            sort_feat_score=feat_score.sort_values('score',ascending=False)
            pool=sort_feat_score['feat'][0:None].tolist()

            clf = svm.LinearSVC(random_state=1).fit(n_x, label_train)
            array3=ch2.transform(vectorizer.transform(x_test))
            test_test_predict = clf.predict(array3)
            acc_x.append(accuracy_score(label_test, test_test_predict))
            k=limit*80/100
            upperlimmit=None
        else:
            pool_2_sub=pool[math.ceil(k):upperlimmit]
            upperlimmit=math.ceil(k)-1

            #feature removal, new set extraction
            new_x_train,new_x_test=bench.transform_features(pool_2_sub,new_x_train,x_test)

            vectorizer = TfidfVectorizer(lowercase=False)
            n_x = vectorizer.fit_transform(new_x_train)

            clf = svm.LinearSVC(random_state=1).fit(n_x, label_train)
            array3=vectorizer.transform(new_x_test)
            test_test_predict = clf.predict(array3)
            acc_x.append(accuracy_score(label_test, test_test_predict))
            k=k/2
    
    df["chi2_acc"]=acc_x
    



    ############ MUTUAL INFORMATION ########################
    new_x_train=bench.x_train
    x_test=bench.x_test


    k=100.2#ignore that
    limit=1000
    acc_x=[]
    while(k>0.001*limit):
        if(k==100.2):
            vectorizer = TfidfVectorizer(lowercase=False)
            n_x = vectorizer.fit_transform(new_x_train)
            limit=len(vectorizer.get_feature_names())
            features=vectorizer.get_feature_names()
            mi = SelectKBest(mutual_info_classif, k="all")
            n_x = mi.fit_transform(n_x, label_train)
            scores=mi.scores_ #get the list of scores and corresponding features

            d = {'feat': features,'score': scores}# make a list of best features regarding mutual_info_classif   
            feat_score = pd.DataFrame(data=d)
            sort_feat_score=feat_score.sort_values('score',ascending=False)
            pool=sort_feat_score['feat'][0:None].tolist()

            clf = svm.LinearSVC(random_state=1).fit(n_x, label_train)
            array3=mi.transform(vectorizer.transform(x_test))
            test_test_predict = clf.predict(array3)
            acc_mi.append(accuracy_score(label_test, test_test_predict))
            k=limit*80/100
            upperlimmit=None
        else:
            pool_2_sub=pool[math.ceil(k):upperlimmit]
            upperlimmit=math.ceil(k)-1

            #feature removal, new set extraction
            new_x_train,new_x_test=bench.transform_features(pool_2_sub,new_x_train,x_test)

            vectorizer = TfidfVectorizer(lowercase=False)
            n_x = vectorizer.fit_transform(new_x_train)

            clf = svm.LinearSVC(random_state=1).fit(n_x, label_train)
            array3=vectorizer.transform(new_x_test)
            test_test_predict = clf.predict(array3)
            acc_mi.append(accuracy_score(label_test, test_test_predict))
            k=k/2
    
    df["mutual_information"]=acc_mi












    df.to_csv('csvs/' + category+'.csv', index=False)        



