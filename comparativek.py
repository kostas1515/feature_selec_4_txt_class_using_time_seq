import os
from FeatureSelection import FeatureSelection
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.feature_extraction.text import  TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, chi2


fig, ax = plt.subplots(1, 1)

bench=FeatureSelection("C",26150) #enter target category and the last id of the preffered train_set
#use relative path
for csv in os.listdir("../testspace2/csvs"):
    data = pd.read_csv("../testspace2/csvs/"+csv, encoding = 'iso-8859-1')
    bench.split_data(data)


label_train=bench.y_train
label_test=bench.y_test
x_test=bench.x_test
x_test_u = list(map(lambda x: str(x), x_test))

n_x_train=bench.rdf(None)
# bench.rdf(topk=1000)
# bench.uniform('single',decision_thres=0.5,topk=1000)
# bench.random_select(1000)
# new_x_train=bench.x_train #for chi squere only


limit=len(bench.rdf_rel_pool)
k=limit
acc=[]
axes=[]
while(k>limit*0.001):
    if(k==limit):
        vectorizer = TfidfVectorizer(lowercase=False)
        n_x = vectorizer.fit_transform(n_x_train)
        clf = svm.LinearSVC().fit(n_x, label_train)
        array3=vectorizer.transform(x_test_u)
        test_test_predict = clf.predict(array3)
        acc.append(accuracy_score(label_test, test_test_predict))
        k=limit*80/100
    else:
        rel_pool=bench.rdf_rel_pool[0:k]

        temp_list=[]
        list2sub=[] #temp list that holds elements to subtract
        new_x_train=[]
        for txt2 in n_x_train:
            temp_list=txt2.split()
            for feature in temp_list:
                if feature not in rel_pool:
                    list2sub.append(feature)
            for x in list2sub:
                temp_list.remove(x)
            list2sub=[]
            str1=' '.join(temp_list)
            temp_list=[]
            new_x_train.append(str1)
        n_x_train=new_x_train
        vectorizer = TfidfVectorizer(lowercase=False)
        n_x = vectorizer.fit_transform(n_x_train)
        clf = svm.LinearSVC().fit(n_x, label_train)
        array3=vectorizer.transform(x_test_u)
        test_test_predict = clf.predict(array3)
        acc.append(accuracy_score(label_test, test_test_predict))
        k=k/2

ax.plot(acc,'bo', label= 'rdf')

ax.legend()
plt.title('Comparison of methods')



################## SINGLE UNIFORM ##############################

n_x_train=bench.uniform('single',decision_thres=0.5,topk=None)
# bench.rdf(topk=1000)
# bench.uniform('single',decision_thres=0.5,topk=1000)
# bench.random_select(1000)
# new_x_train=bench.x_train #for chi squere only

limit=len(bench.uniform_feat_pool)
k=limit
acc=[]
axes=[]
while(k>limit*0.001):
    if(k==limit):
        vectorizer = TfidfVectorizer(lowercase=False)
        n_x = vectorizer.fit_transform(n_x_train)
        clf = svm.LinearSVC().fit(n_x, label_train)
        array3=vectorizer.transform(x_test_u)
        test_test_predict = clf.predict(array3)
        acc.append(accuracy_score(label_test, test_test_predict))
        k=limit*80/100
    else:
        rel_pool=bench.uniform_feat_pool[0:k]

        temp_list=[]
        list2sub=[] #temp list that holds elements to subtract
        new_x_train=[]
        for txt2 in n_x_train:
            temp_list=txt2.split()
            for feature in temp_list:
                if feature not in rel_pool:
                    list2sub.append(feature)
            for x in list2sub:
                temp_list.remove(x)
            list2sub=[]
            str1=' '.join(temp_list)
            temp_list=[]
            new_x_train.append(str1)
        n_x_train=new_x_train
        vectorizer = TfidfVectorizer(lowercase=False)
        n_x = vectorizer.fit_transform(n_x_train)
        clf = svm.LinearSVC().fit(n_x, label_train)
        array3=vectorizer.transform(x_test_u)
        test_test_predict = clf.predict(array3)
        acc.append(accuracy_score(label_test, test_test_predict))
        k=k/2

ax.plot(acc,'ro', label= 'single uniform')


################## SINGLE UNIFORM ##############################

n_x_train=bench.uniform('daily',decision_thres=0.5,topk=None)
# bench.rdf(topk=1000)
# bench.uniform('single',decision_thres=0.5,topk=1000)
# bench.random_select(1000)
# new_x_train=bench.x_train #for chi squere only

limit=len(bench.uniform_feat_pool)
k=limit
acc=[]
axes=[]
while(k>limit*0.001):
    if(k==limit):	
        vectorizer = TfidfVectorizer(lowercase=False)
        n_x = vectorizer.fit_transform(n_x_train)
        clf = svm.LinearSVC().fit(n_x, label_train)
        array3=vectorizer.transform(x_test_u)
        test_test_predict = clf.predict(array3)
        acc.append(accuracy_score(label_test, test_test_predict))
        k=limit*80/100
    else:
        rel_pool=bench.uniform_feat_pool[0:k]

        temp_list=[]
        list2sub=[] #temp list that holds elements to subtract
        new_x_train=[]
        for txt2 in n_x_train:
            temp_list=txt2.split()
            for feature in temp_list:
                if feature not in rel_pool:
                    list2sub.append(feature)
            for x in list2sub:
                temp_list.remove(x)
            list2sub=[]
            str1=' '.join(temp_list)
            temp_list=[]
            new_x_train.append(str1)
        n_x_train=new_x_train
        vectorizer = TfidfVectorizer(lowercase=False)
        n_x = vectorizer.fit_transform(n_x_train)
        clf = svm.LinearSVC().fit(n_x, label_train)
        array3=vectorizer.transform(x_test_u)
        test_test_predict = clf.predict(array3)
        acc.append(accuracy_score(label_test, test_test_predict))
        k=k/2

ax.plot(acc,'mo', label= 'daily uniform')





##################  CHI2 ##############################

new_x_train=bench.x_train
# bench.rdf(topk=1000)
# bench.uniform('single',decision_thres=0.5,topk=1000)
# bench.random_select(1000)
# new_x_train=bench.x_train #for chi squere only

k=100
acc=[]

while(k>1):
    if(k==100):
        vectorizer = TfidfVectorizer(lowercase=False)
        n_x = vectorizer.fit_transform(new_x_train)
        ch2 = SelectPercentile(chi2, percentile=100)
        n_x = ch2.fit_transform(n_x, label_train)
        clf = svm.LinearSVC().fit(n_x, label_train)
        array3=ch2.transform(vectorizer.transform(x_test_u))
        test_test_predict = clf.predict(array3)
        acc.append(accuracy_score(label_test, test_test_predict))
        k=80
    else:
        vectorizer = TfidfVectorizer(lowercase=False)
        n_x = vectorizer.fit_transform(new_x_train)
        ch2 = SelectPercentile(chi2, percentile=k)
        n_x = ch2.fit_transform(n_x, label_train)
        clf = svm.LinearSVC().fit(n_x, label_train)
        array3=ch2.transform(vectorizer.transform(x_test_u))
        test_test_predict = clf.predict(array3)
        acc.append(accuracy_score(label_test, test_test_predict))

        k=k/2

ax.plot(acc,'ko', label= 'chi2')

plt.show()