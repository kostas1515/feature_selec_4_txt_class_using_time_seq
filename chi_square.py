import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

data = pd.read_csv('19960820.csv', encoding = 'iso-8859-1')
pd.set_option('display.max_colwidth', -1) #to see the whole index
pd.set_option('display.max_rows',60)
pd.set_option('display.max_columns', 40)

#initializers
vectorizer = TfidfVectorizer(lowercase= False)

#categories are C E M G
target_category="C"
partition_by_id=3001 # select until what id you split the trai-test



def target_cat(text):
	text=text[:-1]# strip last ;
	array=text.split(';')
	for x in array:
		if (x.startswith(target_category)):
			return 1
	return 0

data['topic_bool']=data['topic'].map(lambda text :target_cat(text))

x_train=[]
y_train=[]
x_test=[]
y_test=[]

for index,row in data.iterrows():
	if ( int(row['filename']) < partition_by_id ):
		x_train.append(row['text'])
		y_train.append(row['topic_bool'])
	else:
		x_test.append(row['text'])
		y_test.append(row['topic_bool'])



X=vectorizer.fit_transform(x_train)
X_new = SelectKBest(chi2, k=200).fit_transform(X, y_train)








