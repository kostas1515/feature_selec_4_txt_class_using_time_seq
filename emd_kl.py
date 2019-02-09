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
category_matrix=["C1","C11","C12","C13","C14","C15","C151","C1511","C152","C16","C17","C171","C172","C173","C174","C18","C181","C182","C183","C2","C21",
"C22","C23","C24","C3","C31","C311","C312","C313","C32","C33","C331","C34","C4","C41","C411","C42","C","E1","E11","E12","E121","E13",
"E131","E132","E14","E141","E142","E143","E2","E21","E211","E212","E3","E31","E311","E312","E313","E4","E41","E411","E5","E51","E511",
"E512","E513","E6","E61","E7","E71","E","G1","G15","G151","G152","G153","G154","G155","G156","G157","G158","G159","G","GCRIM","GDEF","GDIP","GDIS","GENT","GENV","GFAS","GHEA","GJOB","GMIL","GOBIT","GODD","GPOL","GPRO","GREL","GSCI","GSPO","GTOUR",
"GVIO","GVOTE","GWEA","GWELF","M1","M11","M12","M13","M131","M132","M14","M141","M142","M143","M"]

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
    
    x_train,x_test=bench.remove_non_rel_features(x_train,x_test)
    
    threshold=[95,90,85,80,70,50,20,10,5]
    
    
    feature_amount=x_train.shape[1]
    
    uni_order_score_w, uni_stamp_score_w,uni_order_score_kl,uni_stamp_score_kl=bench.emd_and_kl(x_rel_train)

    






    ###################################### Uniform time order ###############################
    p_p=[]
    r_r=[]
    fm=[]

    for thresh in threshold:
        k=math.floor(feature_amount*thresh/100)
        new_x_train,new_x_test=bench.transform_features(x_train,x_test,uni_order_score_w,k)

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
        

    df["uniformO_p_w"]=p_p
    df["uniformO_r_w"]=r_r
    f1["uniformO_f1_w"]=fm


    ############################### UNIFORM TIME STAMP #################################

    p_p=[]
    r_r=[]
    fm=[]

    for thresh in threshold:
        k=math.floor(feature_amount*thresh/100)
        new_x_train,new_x_test=bench.transform_features(x_train,x_test,uni_stamp_score_w,k)

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
        

    df["uniformS_p_w"]=p_p
    df["uniformS_r_w"]=r_r
    f1["uniformS_f1_w"]= fm
    
    
    ###################################### Uniform time order ###############################
    p_p=[]
    r_r=[]
    fm=[]

    for thresh in threshold:
        k=math.floor(feature_amount*thresh/100)
        new_x_train,new_x_test=bench.transform_features(x_train,x_test,uni_order_score_kl,k)

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
        

    df["uniformO_p_kl"]= p_p
    df["uniformO_r_kl"]=r_r
    f1["uniformO_f1_kl"]=fm


    ############################### UNIFORM TIME STAMP #################################

    p_p=[]
    r_r=[]
    fm=[]

    for thresh in threshold:
        k=math.floor(feature_amount*thresh/100)
        new_x_train,new_x_test=bench.transform_features(x_train,x_test,uni_stamp_score_kl,k)

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
        

    df["uniformS_p_kl"]=p_p
    df["uniformS_r_kl"]= r_r
    f1["uniformS_f1_kl"]=fm







    df.to_csv('csvs_revised/p_r_emd_kl/' + category+'.csv', index=False)
    f1.to_csv('csvs_revised/f1_emd_kl/' + category+'.csv', index=False)