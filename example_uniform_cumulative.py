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
target_category="C"
partition_by_id=3001 # select until what id you split the train-testS


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
k=0
relative_k=1 #counter of the step interval 
temp_sum=0
temp_cumulative=[]
max_p_val=0
max_feature=''
for feat in rel_pool:
    while(k<timeline):
	    if feat not in x_rel_train[k].split():
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
    if( max_p_val<p ):
        max_p_val=p
        max_feature=feat
        ttcc=tc
    k=0
    relative_k=1
    temp_sum=0
    temp_cumulative=[]
ax.step(axes,ttcc,'bo', label= 'best feature')
ax.step(axes,cc, 'ro', label= 'optimal distribution')
ax.legend()
plt.xlabel('occurancies')
plt.ylabel('probability')
plt.title('discrete cumulative uniform distribution for E')

plt.show()
# print(sorted(p_val))
d = {'p_val': p_val, 'feat': rel_pool}
df = pd.DataFrame(data=d)
print(df.sort_values(by='p_val', ascending=False)[0:100])#select the first 100 

# print(max_feature)


print("new method")


x_vec=CountVectorizer(lowercase =False)
x_rel_train=x_vec.fit_transform(x_rel_train)






amount_of_documents=x_rel_train.shape[0]
amount_of_features=x_rel_train.shape[1]

#build the optimal uniform function
opt_uni2=np.ones((amount_of_documents,1),dtype=int)
opt_uni2=np.cumsum(opt_uni2)
opt_uni2=opt_uni2/opt_uni2[-1]
print(opt_uni2)
k=0
p_val=[]
while(k<amount_of_features):#check each rel feature what is its distribution 
    arr=x_rel_train[:,k]
    arr=arr.toarray()
    arr=np.cumsum(arr)
    arr=arr/arr[-1]
    p=stats.ks_2samp(opt_uni2,arr)[1]
    p_val.append([p,k])
    k=k+1


final_pval=sorted(p_val, key=lambda x: x[0],reverse =True)[0:100]












