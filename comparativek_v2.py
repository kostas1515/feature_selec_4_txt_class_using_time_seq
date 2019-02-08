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
#category_matrix=["C172","C18","C181","C313","GCRIM","GDIS","GFAS","GJOB","GSCI","GVOTE","GWEA","C183","C22","C32","C411","E13","E212","C173"]
category_matrix=["C172"]
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
    
    
    threshold=[95,90,85,80,70,50,20,10,5]
    feature_amount=x_train.shape[1]
    k=feature_amount

    f_score=[] # array holding the scores for all feat and all_rel feats
    precision=[] # these 3 arrays will have same values for all methods
    recall=[]
    axes=[]
    ######################## No feature selection ##############################
    if (k==feature_amount):
        t_vectorizer = TfidfTransformer()
        x_train_init = t_vectorizer.fit_transform(x_train)

        clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)
        x_test_init=t_vectorizer.transform(x_test)

        test_test_predict = clf.predict(x_test_init)
        precision.append(precision_score(label_test, test_test_predict))
        recall.append(recall_score(label_test, test_test_predict))
        f_score.append(f1_score(label_test, test_test_predict))
        axes.append("all")




    x_rel_train=bench.get_x_rel_train(x_train,label_train)

    rdf_score,uni_order_score,uni_stamp_score=bench.feature_selection(x_rel_train)


    # x_train and x_test contain only relevant features
    x_train,x_test=bench.remove_non_rel_features(x_train,x_test)

    feature_amount=x_train.shape[1]
    k=feature_amount

    ################## Only relevant feature selection #####################
    if (k==feature_amount):
        t_vectorizer = TfidfTransformer()
        x_train_init = t_vectorizer.fit_transform(x_train)

        clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)
        x_test_init=t_vectorizer.transform(x_test)

        test_test_predict = clf.predict(x_test_init)
        precision.append(precision_score(label_test, test_test_predict))
        recall.append(recall_score(label_test, test_test_predict))
        f_score.append(f1_score(label_test, test_test_predict))
        axes.append("rel_feat")




    ############################ RDF FEATURES ######################################
    p_p=[]
    r_r=[]
    fm=[]
    for thresh in threshold:
        k=math.floor(feature_amount*thresh/100)
        new_x_train,new_x_test=bench.transform_features(x_train,x_test,rdf_score,k)

            #tfidf
        t_vectorizer = TfidfTransformer()
        x_train_init = t_vectorizer.fit_transform(new_x_train)
        x_test_init= t_vectorizer.transform(new_x_test)
        
        #classification

        clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)
        test_test_predict = clf.predict(x_test_init)


        p_p.append(precision_score(label_test, test_test_predict))
        r_r.append(recall_score(label_test, test_test_predict))
        fm.append(f1_score(label_test, test_test_predict))
        axes.append(thresh)

    df["axes"]=axes
    f1["axes"]=axes
    df["rdf_p"]=precision + p_p
    df["rdf_r"]=recall + r_r
    f1["rdf_f1"]=f_score + fm



    ###################################### Uniform time order ###############################
    p_p=[]
    r_r=[]
    fm=[]

    for thresh in threshold:
        k=math.floor(feature_amount*thresh/100)
        new_x_train,new_x_test=bench.transform_features(x_train,x_test,uni_order_score,k)

            #tfidf
        t_vectorizer = TfidfTransformer()
        x_train_init = t_vectorizer.fit_transform(new_x_train)
        x_test_init= t_vectorizer.transform(new_x_test)
        
        #classification

        clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)
        test_test_predict = clf.predict(x_test_init)


        p_p.append(precision_score(label_test, test_test_predict))
        r_r.append(recall_score(label_test, test_test_predict))
        fm.append(f1_score(label_test, test_test_predict))
        

    df["uniformO_p"]=precision + p_p
    df["uniformO_r"]=recall + r_r
    f1["uniformO_f1"]=f_score + fm


    ############################### UNIFORM TIME STAMP #################################

    p_p=[]
    r_r=[]
    fm=[]

    for thresh in threshold:
        k=math.floor(feature_amount*thresh/100)
        new_x_train,new_x_test=bench.transform_features(x_train,x_test,uni_stamp_score,k)

            #tfidf
        t_vectorizer = TfidfTransformer()
        x_train_init = t_vectorizer.fit_transform(new_x_train)
        x_test_init= t_vectorizer.transform(new_x_test)
        
        #classification

        clf = svm.LinearSVC(random_state=1).fit(x_train_init, label_train)
        test_test_predict = clf.predict(x_test_init)


        p_p.append(precision_score(label_test, test_test_predict))
        r_r.append(recall_score(label_test, test_test_predict))
        fm.append(f1_score(label_test, test_test_predict))
        

    df["uniformS_p"]=precision + p_p
    df["uniformS_r"]=recall + r_r
    f1["uniformS_f1"]=f_score + fm





    ##################  CHI2 ##############################

    p_p=[]
    r_r=[]
    fm=[]

    for thresh in threshold:
        k=math.floor(feature_amount*thresh/100)
        ch2 = SelectKBest(chi2, k=k)
        new_x_train_init = ch2.fit_transform(x_train, label_train)

        t_vectorizer = TfidfTransformer()
        new_x_train_init=t_vectorizer.fit_transform(new_x_train_init)

        clf = svm.LinearSVC(random_state=1).fit(new_x_train_init, label_train)

        new_x_test_init=ch2.transform(x_test)
        new_x_test_init=t_vectorizer.transform(new_x_test_init)

        test_test_predict = clf.predict(new_x_test_init)

        p_p.append(precision_score(label_test, test_test_predict))
        r_r.append(recall_score(label_test, test_test_predict))
        fm.append(f1_score(label_test, test_test_predict))
    
    df["chi2_p"]=precision + p_p
    df["chi2_r"]=recall + r_r
    f1["chi2_f1"]=f_score + fm
    



    ############ MUTUAL INFORMATION ########################

    p_p=[]
    r_r=[]
    fm=[]

    for thresh in threshold:
        k=math.floor(feature_amount*thresh/100)
        mi = SelectKBest(mutual_info_classif, k=k)
        new_x_train_init = mi.fit_transform(x_train, label_train)

        t_vectorizer = TfidfTransformer()
        new_x_train_init=t_vectorizer.fit_transform(new_x_train_init)

        clf = svm.LinearSVC(random_state=1).fit(new_x_train_init, label_train)

        new_x_test_init=mi.transform(x_test)
        new_x_test_init=t_vectorizer.transform(new_x_test_init)


        test_test_predict = clf.predict(new_x_test_init)

        p_p.append(precision_score(label_test, test_test_predict))
        r_r.append(recall_score(label_test, test_test_predict))
        fm.append(f1_score(label_test, test_test_predict))
    
    
    df["mutual_info_p"]=precision + p_p
    df["mutual_info_r"]=recall + r_r
    f1["mutual_info_f1"]=f_score + fm



    df.to_csv('csvs_revised/p_r/' + category+'.csv', index=False)
    f1.to_csv('csvs_revised/f1/' + category+'.csv', index=False)




