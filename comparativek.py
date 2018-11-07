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


category_matrix=["C","C1","C11","C12","C13","C14","C15","C151","C152","C16","C17","C171","C172","C173","C174","C18","C181","C182",
"C183","C2","C21","C22","C23","C24","C31","C311","C312","C313","C32","C33","C331","C34","C3","C4","C41","C411","C42","E","E1","E11","E12","E121","E13","E131","E132","E14","E141","E142","E143",
"E2","E21","E211","E212","E3","E311","E312","E313","E4","E41","E411","E5","E51","E511","E512","E513","E6","E61","E7","E71""G","G1","G15","G151","G152","G153","G154","G156","G157","G158","G159"
"GCRIM","GDEF","GDIP","GDIS","GENT","GENV","GHEA","GJOB","GOBIT","GODD","GPOL","GMIL","GREL","GTOUR","GELF","GFAS","GPRO","GSCI","GSPO","GVIO","GVOTE","GWEA","M","M11","M12","M13","M131","M132","M14","M141","M142","M143"]
# category_matrix=["C"]
for category in category_matrix:
    df=pd.DataFrame()

    bench=FeatureSelection(category,26150) #enter target category and the last id of the preffered train_set
    #use relative path
    for csv in os.listdir("../testspace2/csvs"):
        data = pd.read_csv("../testspace2/csvs/"+csv, encoding = 'iso-8859-1')
        bench.split_data(data)

    label_train=bench.y_train
    if (sum(label_train)>len(label_train)*10/100):  # the train set should have at least 10% of target class

        vectorizer = CountVectorizer(lowercase=False)
        x_train = vectorizer.fit_transform(bench.x_train)
        x_test=vectorizer.transform(bench.x_test)
        
        label_test=bench.y_test
        



        score=bench.quick_rdf(x_train,label_train)

        feature_amount=len(score)
        k=feature_amount
        acc_r=[]
        axes=[]
        percent=80
        while(k>feature_amount*0.001):
            if(k==feature_amount):

                t_vectorizer = TfidfTransformer()
                x_train_init = t_vectorizer.fit_transform(x_train)

                clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)
                x_test_init=t_vectorizer.transform(x_test)

                test_test_predict = clf.predict(x_test_init)
                acc_r.append(accuracy_score(label_test, test_test_predict))

                k=math.floor(feature_amount*80/100)
                axes.append(100)
            else:

                new_x_train,new_x_test=bench.transform_features(x_train_init,x_test_init,score,k)
                
                #classification

                clf = svm.LinearSVC(random_state=1).fit(new_x_train, label_train)
                test_test_predict = clf.predict(new_x_test)


                acc_r.append(accuracy_score(label_test, test_test_predict))
                axes.append(percent)
                percent=percent/2
                k=math.floor(k/2)

        df["axes"]=axes
        df["rdf_acc"]=acc_r



        



        ################## SINGLE UNIFORM ##############################

        # bench.rdf(topk=1000)
        # bench.uniform('single',decision_thres=0.5,topk=1000)
        # bench.random_select(1000)
        # new_x_train=bench.x_train #for chi squere only

        score=bench.quick_uniform(x_train,label_train)

        feature_amount=len(score)
        k=feature_amount
        acc_u=[]
        axes=[]
        percent=80
        while(k>feature_amount*0.001):
            if(k==feature_amount):

                t_vectorizer = TfidfTransformer()
                x_train_init = t_vectorizer.fit_transform(x_train)
                clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)

                x_test_init=t_vectorizer.transform(x_test)
                test_test_predict = clf.predict(x_test_init)
                acc_u.append(accuracy_score(label_test, test_test_predict))

                k=math.floor(feature_amount*80/100)
            else:
                new_x_train,new_x_test=bench.transform_features(x_train_init,x_test_init,score,k)
                
                #classification
                
                clf = svm.LinearSVC(random_state=1).fit(new_x_train, label_train)
                test_test_predict = clf.predict(new_x_test)


                acc_u.append(accuracy_score(label_test, test_test_predict))
                k=math.floor(k/2)

        df["uniform_acc"]=acc_u


        ##################  CHI2 ##############################

        x_train=bench.x_train
        x_test=bench.x_test

        k=100.2#ignore that
        limit=1000
        acc_x=[]
        while(k>0.001*limit):
            if(k==100.2):
                vectorizer = TfidfVectorizer(lowercase=False)
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

                finalscore=sorted(finalscore,key=lambda x: x[0],reverse =True)
                

                clf = svm.LinearSVC(random_state=1).fit(x_train, label_train)
                x_test=ch2.transform(vectorizer.transform(x_test))

                test_test_predict = clf.predict(x_test)
                acc_x.append(accuracy_score(label_test, test_test_predict))
                k=math.floor(limit*80/100)
            else:
                new_x_train,new_x_test=bench.transform_features(x_train,x_test,finalscore,k)

                clf = svm.LinearSVC(random_state=1).fit(new_x_train, label_train)


                test_test_predict = clf.predict(new_x_test)
                acc_x.append(accuracy_score(label_test, test_test_predict))
                k=math.floor(k/2)
        
        df["chi2_acc"]=acc_x
        



        ############ MUTUAL INFORMATION ########################
        x_train=bench.x_train
        x_test=bench.x_test

        k=100.2#ignore that
        limit=1000
        acc_m=[]
        while(k>0.001*limit):
            if(k==100.2):
                vectorizer = TfidfVectorizer(lowercase=False)
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

                finalscore=sorted(finalscore,key=lambda x: x[0],reverse =True)
                

                clf = svm.LinearSVC(random_state=1).fit(x_train, label_train)
                x_test=mi.transform(vectorizer.transform(x_test))

                test_test_predict = clf.predict(x_test)
                acc_m.append(accuracy_score(label_test, test_test_predict))
                k=math.floor(limit*80/100)
            else:
                new_x_train,new_x_test=bench.transform_features(x_train,x_test,finalscore,k)

                clf = svm.LinearSVC(random_state=1).fit(new_x_train, label_train)


                test_test_predict = clf.predict(new_x_test)
                acc_m.append(accuracy_score(label_test, test_test_predict))
                k=math.floor(k/2)
        
        df["mutual_info_acc"]=acc_m












        df.to_csv('csvs/' + category+'.csv', index=False)        



