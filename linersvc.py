import os
from FeatureSelection import FeatureSelection
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.feature_extraction.text import  TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import coo_matrix, hstack

bench=FeatureSelection("GSPO",26150) #enter target category and the last id of the preffered train_set
#use relative path
for csv in os.listdir("../testspace2/csvs"):
	data = pd.read_csv("../testspace2/csvs/"+csv, encoding = 'iso-8859-1')
	bench.split_data(data)

# integrate different features
def add_row(array1,array2):
	array2=np.array(array2).reshape(len(array2),1)
	array3=hstack((array1,array2))
	return array3



new_x_train,new_x_test=bench.rdf(topk=1000)
# bench.rdf(topk=1000)
# bench.uniform('single',decision_thres=0.5,topk=1000)
# bench.random_select(1000)
# new_x_train=bench.x_train #for chi squere only


label_train=bench.y_train
label_test=bench.y_test


vectorizer = TfidfVectorizer(lowercase=False)
n_x = vectorizer.fit_transform(new_x_train)

#uncomment the 2 lines below fo chi2 
# ch2 = SelectKBest(chi2, k=1000)
# n_x = ch2.fit_transform(n_x, label_train)


#TRAIN PHASE
n_x=add_row(n_x,bench.is_wknd_train)
clf = svm.LinearSVC(random_state=1).fit(n_x, label_train)






#TEST PHASEz
#uncomment the line below for chi2,comment the next 
# array3=ch2.transform(vectorizer.transform(x_test))



array3=vectorizer.transform(new_x_test)
array3=add_row(array3,bench.is_wknd_test)

test_test_predict = clf.predict(array3)


conf_test = confusion_matrix(label_test, test_test_predict)


print(clf)
print("\nTest\n")
 
print ('accuracy', accuracy_score(label_test, test_test_predict))
print ('confusion matrix\n', confusion_matrix(label_test, test_test_predict))
print ('(row=expected, col=predicted)')
print(classification_report(label_test, test_test_predict))

plt.figure()
bench.plot_confusion_matrix(conf_test, classes="",
                      title='Confusion matrix, for class '+ bench.target_cat )

plt.show()