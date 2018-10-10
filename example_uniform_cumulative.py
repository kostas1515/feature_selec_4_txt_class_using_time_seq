from scipy import stats
import pandas as pd
from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import SelectKBest 

data = pd.read_csv('19960820.csv', encoding = 'iso-8859-1')
pd.set_option('display.max_colwidth', -1) #to see the whole index
pd.set_option('display.max_rows',-1)
pd.set_option('display.max_columns', 40)
fig, ax = plt.subplots(1, 1)


#categories are C E M G
target_category="G"
partition_by_id=3001 # select until what id you split the train-test


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



# make an only rel sub matrix of x_train
# subtrack the non-rel
x_rel_train=[]
for txt,is_rel in zip(x_train,y_train):
	if (is_rel==1):
		x_rel_train.append(txt)


rel_vectorizer=CountVectorizer(lowercase =False)
rel_vectorizer.fit_transform(x_rel_train)
rel_pool=rel_vectorizer.get_feature_names() #this is the relative features pool in alphabetic order

timeline=sum(y_train) # the sum of ralative documents
print(timeline)
step=1/timeline

#first build the optimal cumulative uniform discrete function
cumulative=[]
axes=[]
k=1 #counter
while(k<=timeline):
	cumulative.append(step*k)
	axes.append(k-1)
	k=k+1

#calculate the cumulative distribution for every feature and get the p value
p_val=[]
k=1
relative_k=1 #counter of the step interval 
temp_sum=0
temp_cumulative=[]
max_p_val=0
max_feature=''
feat=rel_pool[10]
while(k<=timeline):
	if feat not in x_rel_train[k-1].split():
		temp_cumulative.append(temp_sum)
	else:
		temp_cumulative.append(relative_k)
		temp_sum=relative_k
		relative_k=relative_k+1
	k=k+1

tc=np.array(temp_cumulative)/(relative_k-1)
cc=np.array(cumulative)
p=stats.ks_2samp(cc, tc)[1]
p_val.append(p)
# if( max_p_val<p ):
# 	max_p_val=p
# 	max_feature=feat
# temp_cumulative=[]
# k=1
ax.step(axes,cc, 'ro')
ax.step(axes,tc,'bo')
plt.show()

# d = {'p_val': p_val, 'feat': rel_pool}
# df = pd.DataFrame(data=d)
# new_df=df.sort_values(by='p_val', ascending=False)[0:100]#select the first 100 

