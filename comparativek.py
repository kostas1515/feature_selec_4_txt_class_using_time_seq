import os
from FeatureSelection import FeatureSelection
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix,precision_score,recall_score
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.feature_extraction.text import  TfidfVectorizer,TfidfTransformer,CountVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import math

# row['percent_train']>0.02
category_matrix=["C1","C11","C13","C15","C151","C1511","C152","C17","C171","C18","C181","C2","C21","C24","C3","C31","C4","C","E1","E12","E2","E21","E212","E4","E41","E5","E51","E","G1","G15","G","GCRIM","GDIP","GJOB","GPOL","GSPO","GVIO","M1","M11","M12","M13","M131","M132","M14","M141","M143","M"]
for category in category_matrix:
    df=pd.DataFrame()
    f1=pd.DataFrame()

    bench=FeatureSelection(category,389827) #enter target category and the last id of the preffered train_set
    #use relative path
    for csv in os.listdir("../testspace2/csvs"):
        data = pd.read_csv("../testspace2/csvs/"+csv, encoding = 'iso-8859-1')
        bench.split_data(data)

    label_train=bench.y_train
    vectorizer = CountVectorizer(lowercase=False)
    x_train = vectorizer.fit_transform(bench.x_train)
    x_test=vectorizer.transform(bench.x_test)
    
    label_test=bench.y_test
    


    x_rel_train=bench.get_x_rel_train(x_train,label_train)

    score=bench.quick_rdf(x_rel_train)

    feature_amount=len(score)
    k=feature_amount
    fm=[]
    p_r=[]
    r_r=[]
    axes=[]
    percent=50
    while(k>feature_amount*0.01):
        if(k==feature_amount):

            t_vectorizer = TfidfTransformer()
            x_train_init = t_vectorizer.fit_transform(x_train)

            clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)
            x_test_init=t_vectorizer.transform(x_test)

            test_test_predict = clf.predict(x_test_init)
            p_r.append(precision_score(label_test, test_test_predict))
            r_r.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))

            k=math.floor(feature_amount*50/100)
            axes.append(100)
        else:

            new_x_train,new_x_test=bench.transform_features(x_train_init,x_test_init,score,k)
            
            #classification

            clf = svm.LinearSVC(random_state=1).fit(new_x_train, label_train)
            test_test_predict = clf.predict(new_x_test)


            p_r.append(precision_score(label_test, test_test_predict))
            r_r.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))

            axes.append(percent)
            percent=percent/2
            k=math.floor(k/2)

    df["axes"]=axes
    f1["axes"]=axes
    df["rdf_p"]=p_r
    df["rdf_r"]=r_r
    f1["rdf_f1"]=fm




    



    ################## SINGLE UNIFORM ##############################

    # bench.rdf(topk=1000)
    # bench.uniform('single',decision_thres=0.5,topk=1000)
    # bench.random_select(1000)
    # new_x_train=bench.x_train #for chi squere only

    score=bench.quick_uniform(x_rel_train)

    feature_amount=len(score)
    k=feature_amount
    p_u=[]
    r_u=[]
    fm=[]
    while(k>feature_amount*0.01):
        if(k==feature_amount):

            t_vectorizer = TfidfTransformer()
            x_train_init = t_vectorizer.fit_transform(x_train)
            clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)

            x_test_init=t_vectorizer.transform(x_test)
            test_test_predict = clf.predict(x_test_init)
            p_u.append(precision_score(label_test, test_test_predict))
            r_u.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))

            k=math.floor(feature_amount*50/100)
        else:
            new_x_train,new_x_test=bench.transform_features(x_train_init,x_test_init,score,k)
            
            #classification
            
            clf = svm.LinearSVC(random_state=1).fit(new_x_train, label_train)
            test_test_predict = clf.predict(new_x_test)


            p_u.append(precision_score(label_test, test_test_predict))
            r_u.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))

            k=math.floor(k/2)

    df["uniformO_p"]=p_u
    df["uniformO_r"]=r_u
    f1["uniformO_f1"]=fm


    score=bench.quick_uniform2(x_rel_train)

    feature_amount=len(score)
    k=feature_amount
    p_u=[]
    r_u=[]
    fm=[]

    while(k>feature_amount*0.01):
        if(k==feature_amount):

            t_vectorizer = TfidfTransformer()
            x_train_init = t_vectorizer.fit_transform(x_train)
            clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)

            x_test_init=t_vectorizer.transform(x_test)
            test_test_predict = clf.predict(x_test_init)

            p_u.append(precision_score(label_test, test_test_predict))
            r_u.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))

            k=math.floor(feature_amount*50/100)
        else:
            new_x_train,new_x_test=bench.transform_features(x_train_init,x_test_init,score,k)
            
            #classification
            
            clf = svm.LinearSVC(random_state=1).fit(new_x_train, label_train)
            test_test_predict = clf.predict(new_x_test)


            p_u.append(precision_score(label_test, test_test_predict))
            r_u.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))
            k=math.floor(k/2)

    df["uniformS_p"]=p_u
    df["uniformS_r"]=r_u
    f1["uniformS_f1"]=fm


    ##################  CHI2 ##############################

    x_train=bench.x_train
    x_test=bench.x_test

    k=100.2#ignore that
    limit=1000
    p_x=[]
    r_x=[]
    fm=[]
    while(k>0.01*limit):
        if(k==100.2):
            vectorizer = CountVectorizer(lowercase=False)
            x_train = vectorizer.fit_transform(x_train)


            ch2 = SelectKBest(chi2, k="all")
            x_train = ch2.fit_transform(x_train, label_train)

            score=ch2.scores_ #get the list of scores and corresponding features
            limit=len(score)
            finalscore=[]   # get the corresponding index of the feauter and the score
            index=0
            for s in score:
                finalscore.append([s,index])
                index=index+1
            
            for x in bench.list_2_zero: #this code makes all features that occour in less than 5% of total rel docs zero
                finalscore[x]=[0,x]

            finalscore=sorted(finalscore,key=lambda x: x[0],reverse =True)
            
            t_vectorizer = TfidfTransformer()
            x_train=t_vectorizer.fit_transform(x_train)

            clf = svm.LinearSVC(random_state=1).fit(x_train, label_train)


            x_test=ch2.transform(vectorizer.transform(x_test))
            x_test=t_vectorizer.transform(x_test)

            test_test_predict = clf.predict(x_test)

            p_x.append(precision_score(label_test, test_test_predict))
            r_x.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))

            k=math.floor(limit*50/100)
        else:
            new_x_train,new_x_test=bench.transform_features(x_train,x_test,finalscore,k)

            clf = svm.LinearSVC(random_state=1).fit(new_x_train, label_train)


            test_test_predict = clf.predict(new_x_test)

            p_x.append(precision_score(label_test, test_test_predict))
            r_x.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))

            k=math.floor(k/2)
    
    df["chi2_p"]=p_x
    df["chi2_r"]=r_x
    f1["chi2_f1"]=fm
    



    ############ MUTUAL INFORMATION ########################
    x_train=bench.x_train
    x_test=bench.x_test

    k=100.2#ignore that
    limit=1000
    p_m=[]
    r_m=[]
    fm=[]
    while(k>0.01*limit):
        if(k==100.2):
            vectorizer = CountVectorizer(lowercase=False)
            x_train = vectorizer.fit_transform(x_train)


            mi = SelectKBest(mutual_info_classif, k="all")
            x_train = mi.fit_transform(x_train, label_train)

            score=mi.scores_ #get the list of scores and corresponding features
            limit=len(score)
            finalscore=[]   # get the corresponding index of the feauter and the score
            index=0
            for s in score:
                finalscore.append([s,index])
                index=index+1

            for x in bench.list_2_zero: #this code makes all features that occour in less than 5% of total rel docs zero
                finalscore[x]=[0,x]

            finalscore=sorted(finalscore,key=lambda x: x[0],reverse =True)
            
            t_vectorizer = TfidfTransformer()
            x_train=t_vectorizer.fit_transform(x_train)

            clf = svm.LinearSVC(random_state=1).fit(x_train, label_train)
            x_test=mi.transform(vectorizer.transform(x_test))

            x_test=t_vectorizer.transform(x_test)

            test_test_predict = clf.predict(x_test)

            p_m.append(precision_score(label_test, test_test_predict))
            r_m.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))

            k=math.floor(limit*50/100)
        else:
            new_x_train,new_x_test=bench.transform_features(x_train,x_test,finalscore,k)

            clf = svm.LinearSVC(random_state=1).fit(new_x_train, label_train)


            test_test_predict = clf.predict(new_x_test)

            p_m.append(precision_score(label_test, test_test_predict))
            r_m.append(recall_score(label_test, test_test_predict))
            fm.append(f1_score(label_test, test_test_predict))

            k=math.floor(k/2)
    
    df["mutual_info_p"]=p_m
    df["mutual_info_r"]=r_m
    f1["mutual_info_f1"]=fm



    df.to_csv('csvs/p_r/' + category+'.csv', index=False)
    f1.to_csv('csvs/f1/' + category+'.csv', index=False)




