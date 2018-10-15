import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

data = pd.read_csv('19960820.csv', encoding = 'iso-8859-1')
pd.set_option('display.max_colwidth', -1) #to see the whole index
pd.set_option('display.max_rows',60)
pd.set_option('display.max_columns', 40)

#initializers
# vectorizer = TfidfVectorizer(lowercase= False)


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

# print("the " + target_category + " category has "+ str(sum(y_train)) + " documents")
# print("before the non-relevant feature subtraction:")


#get the original pool of terms
# init_vectorizer=CountVectorizer()
# init_vectorizer.fit_transform(x_train)
# init_pool=init_vectorizer.get_feature_names()


# print("the initial pool is " + str(len(init_pool)))
# # print("after the rdf feature selection:")




# make an only rel sub matrix of x_train
# subtrack the non-rel
x_rel_train=[]

for txt,is_rel in zip(x_train,y_train):
	if (is_rel==1):
		x_rel_train.append(txt)


rel_vectorizer=CountVectorizer(lowercase =False)
rel_vectorizer.fit_transform(x_rel_train)
rel_pool=rel_vectorizer.get_feature_names() #this is the relative features pool in alphabetic order


k=0
term_count=0
term_score=[]
for term in rel_pool:
	while(k<len(x_rel_train)):
		if(term in x_rel_train[k].split()):
			term_count=term_count+1
		k=k+1
	term_score.append(term_count)
	term_count=0
	k=0

d = {'feat': rel_pool,'score': term_score}
rdf_feat_score = pd.DataFrame(data=d)

sort_rdf_feat_score=rdf_feat_score.sort_values('score',ascending=False)
rdf_rel_pool=sort_rdf_feat_score['feat'][0:1000]


# print ("the pool of relevant terms has  " + str(len(rel_pool)) +" features.")


# subtrack from x_train all other features that are not included with regards to rel_pool
temp_list=[]
new_x_train=[] #revised train set with only rel terms 
list2sub=[] #temp list that holds elements to subtract

for txt2 in x_train:
	temp_list=txt2.split()
	for feat in temp_list:
		if feat not in rdf_rel_pool:
			list2sub.append(feat)
	for x in list2sub:
		temp_list.remove(x)
	list2sub=[]
	str1=' '.join(temp_list)
	new_x_train.append(str1)



