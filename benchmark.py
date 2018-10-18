import os
from FeatureSelection import FeatureSelection
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.feature_extraction.text import  TfidfVectorizer
import matplotlib.pyplot as plt


bench=FeatureSelection("C",3001) #enter target category and the last id of the preffered train_set
#use relative path
for csv in os.listdir("../testspace2/csvs"):
	data = pd.read_csv("../testspace2/csvs/"+csv, encoding = 'iso-8859-1')
	bench.split_data(data)


uniform_new_x_train=bench.uniform('single',decision_thres=0.5,topk=1000)
# bench.rdf(topk=1000)
# rdf_new_x_train=bench.new_x_train
# bench.chi_squere(1000)
# chi2_new_x_train=bench.new_x_train



label_train=bench.y_train
label_test=bench.y_test
x_test=bench.x_test

vectorizer = TfidfVectorizer(lowercase=False).fit(uniform_new_x_train)

u_x = vectorizer.transform(uniform_new_x_train)
#r_x = vectorizer.fit_transform(rdf_new_x_train)
#c_x = vectorizer.fit_transform(chi2_new_x_train)
#TRAIN PHASE

clf = svm.LinearSVC().fit(u_x, label_train)



#TEST PHASE


array3=vectorizer.transform(x_test)
test_test_predict = clf.predict(array3)


conf_test = confusion_matrix(label_test, test_test_predict)


print(clf)
print("Testttttt\n")
 
print ('accuracy', accuracy_score(label_test, test_test_predict))
print ('confusion matrix\n', confusion_matrix(label_test, test_test_predict))
print ('(row=expected, col=predicted)')
print(classification_report(label_test, test_test_predict))

plt.figure()
bench.plot_confusion_matrix(conf_test, classes="",
                      title='Confusion matrix, for class '+ bench.target_cat )

plt.show()