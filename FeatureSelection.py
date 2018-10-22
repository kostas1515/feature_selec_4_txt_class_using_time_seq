class FeatureSelection():
    def __init__(self,target_cat,partition_by_id,):
        from sklearn.feature_extraction.text import CountVectorizer
        import numpy as np
        from scipy import stats as st
        import pandas as pd 
        import matplotlib.pyplot as plt
        self.plt=plt
        self.pd=pd
        self.st=st
        self.cv=CountVectorizer(lowercase =False)
        self.np=np
        self.partition_by_id=partition_by_id # must be integer 
        self.target_cat = target_cat #must be CCAT OR GCAT GFAS GWEA etc
        self.x_train=[]
        self.y_train=[]
        self.x_test=[]
        self.y_test=[]
        self.file_per_day_array=[]# an array that contains the number of files that day
        self.new_x_train=[]



    def split_data(self,data): #put this method under the open csv loop data must be in panda form
        self.data= data
        self.data['topic_bool']=self.data['topic'].map(lambda text :self.__target_category(text))
        filecounter=0 #this helps the uniform method to define the step interval of uniformity it can be a file or a day (maybe MONTH ???)
        for index,row in self.data.iterrows():
            if ( int(row['filename']) < self.partition_by_id ):
                self.x_train.append(row['text'])
                self.y_train.append(row['topic_bool'])
                if(row['topic_bool']==1):
                        filecounter=filecounter+1
            else:
                self.x_test.append(row['text'])
                self.y_test.append(row['topic_bool'])
        self.file_per_day_array.append(filecounter)
       #returns the x_train y_train x_test and y_test    


    def __target_category(self,text):
        text=text[:-1]# strip last ;
        array=text.split(';')
        for x in array:
            if (x.startswith(self.target_cat)):
                return 1
        return 0
        #this is a private method that heleps the target_category method and transforms the target_cat into 0 ,1


    def uniform(self,step_interval,decision_thres,topk):
        self.topk=topk
        self.step_interval=step_interval #step interval can be either 'single' or day
        self.x_rel_train=[]
        self.decision_thres=decision_thres #it can be from 1 to 0.001 --? 1 means even one occurence count as uniform
        for txt,is_rel in zip(self.x_train,self.y_train): #take only relative
            if (is_rel==1):
                self.x_rel_train.append(txt)


        rel_vectorizer=self.cv
        rel_vectorizer.fit_transform(self.x_rel_train)
        rel_pool=rel_vectorizer.get_feature_names() #this is the relative features pool in alphabetic order

        if (self.step_interval=='single'):
            timeline=sum(self.y_train) # the sum of relative documents
            step=1/timeline

            #first build the optimal cumulative uniform discrete function
            cumulative=[]
            #axes=[]
            k=1 #counter
            while(k<=timeline):
                cumulative.append(step*k)
                #axes.append(k-1)
                k=k+1

            #calculate the cumulative distribution for every feature and get the p value
            p_val=[]
            k=0
            relative_k=1 #counter of the step interval 
            temp_sum=0
            temp_cumulative=[]
            for feat in rel_pool:
                while(k<timeline):
                    if feat not in self.x_rel_train[k].split():
                        temp_cumulative.append(temp_sum)
                    else:
                        temp_cumulative.append(relative_k)
                        temp_sum=relative_k
                        relative_k=relative_k+1
                    k=k+1
                tc=self.np.array(temp_cumulative)/(relative_k-1) #subtract 1 because the last iteration of relative_k would be faulty
                cc=self.np.array(cumulative)
                p=self.st.ks_2samp(cc, tc)[1]
                p_val.append(p)
                temp_cumulative=[]
                temp_sum=0
                k=0
                relative_k=1
        else: #this means that the interval will be a day
            timeline=len(self.file_per_day_array)
            step=1/timeline
            #first build the optimal cumulative uniform discrete function
            cumulative=[]
            axes=[]
            k=1 #counter
            while(k<=timeline):
                cumulative.append(step*k)
                # axes.append(k-1)
                k=k+1

            #calculate the cumulative distribution for every feature and get the p value
            p_val=[]
            k=0 #k goes from zero to timeline
            rel_file=0 #rel_file goes from zero to value of file_per_day_array[k]
            file=0 #file goes from zero to len(x_rel_train)
            temp_sum=0
            temp_cumulative=[]
            tc_count=0 #this is a counter that helps to build the cumulative uniform
            for feat in rel_pool:
                while(k<timeline):
                    while(rel_file<self.file_per_day_array[k]):
                        if feat in self.x_rel_train[rel_file+file].split():
                            temp_sum=temp_sum+1
                        rel_file=rel_file +1
                    if (temp_sum>(self.file_per_day_array[k]*self.decision_thres)):# if the total sum of relevant files containing the feat is higher than the total rel files then consider that feat uniform for that day
                        temp_cumulative.append(1)
                    else:
                        temp_cumulative.append(0)
                    temp_sum=0
                    file=rel_file+file
                    rel_file=0#set this to zero to count the next num of files_per day
                    k=k+1
                # print(sum(temp_cumulative))
                if(sum(temp_cumulative)!=0):
                    tc=self.np.array(temp_cumulative)/(sum(temp_cumulative))
                    while(tc_count<len(tc)-1):
                        tc[tc_count+1]=tc[tc_count]+tc[tc_count+1]
                        tc_count=tc_count+1
                    cc=self.np.array(cumulative)
                    p=self.st.ks_2samp(cc, tc)[1]
                    p_val.append(p)
                else:
                    p_val.append(0)
                temp_cumulative=[]
                temp_sum=0
                k=0
                file=0
                tc_count=0

        d = {'p_val': p_val,'feat': rel_pool}
        p_val_feat = self.pd.DataFrame(data=d) #this is a dataframe containing the values and the features
        sort_p_val_feat=p_val_feat.sort_values('p_val',ascending=False)
        self.uniform_feat_pool=sort_p_val_feat['feat'][0:self.topk].tolist() # uniform_rel_pool was dataframe and it had to be a list
        temp_list=[]
        self.new_x_train=[] #revised train set with only rel terms 
        list2sub=[] #temp list that holds elements to subtract

        for txt2 in self.x_train:
            temp_list=txt2.split()
            for feature in temp_list:
                if feature not in self.uniform_feat_pool:
                    list2sub.append(feature)
            for x in list2sub:
                temp_list.remove(x)
            list2sub=[]
            str1=' '.join(temp_list)
            temp_list=[]
            self.new_x_train.append(str1)
        return self.new_x_train


            


            # ax.step(cc, 'ro')
            # ax.step(tc,'bo')
            # self.plt.show()

    def rdf(self,topk): # it uses Coundvectorizer 
        from sklearn.feature_extraction.text import CountVectorizer
        self.topk=topk
        self.x_rel_train=[]
        for txt,is_rel in zip(self.x_train,self.y_train): #take only relative
            if (is_rel==1):
                self.x_rel_train.append(txt)

        rel_vectorizer=CountVectorizer(lowercase =False)
        rel_vectorizer.fit_transform(self.x_rel_train)
        rel_pool=rel_vectorizer.get_feature_names() #this is the relative features pool in alphabetic order
        
        k=0
        term_count=0
        term_score=[]
        for term in rel_pool:
            while(k<len(self.x_rel_train)):
                if(term in self.x_rel_train[k].split()):
                    term_count=term_count+1
                k=k+1
            term_score.append(term_count)
            term_count=0
            k=0

        d = {'feat': rel_pool,'score': term_score}
        rdf_feat_score = self.pd.DataFrame(data=d)

        sort_rdf_feat_score=rdf_feat_score.sort_values('score',ascending=False)
        rdf_rel_pool=sort_rdf_feat_score['feat'][0:self.topk].tolist() # attention transform the dataframe to a list !!!!


       # print ("the pool of relevant terms has  " + str(len(rel_pool)) +" features.")
       # subtrack from x_train all other features that are not included with regards to rel_pool
        temp_list=[]
        self.new_x_train=[] #revised train set with only rel terms 
        list2sub=[] #temp list that holds elements to subtract

        for txt2 in self.x_train:
            temp_list=txt2.split()
            for feature in temp_list:
                if feature not in rdf_rel_pool:
                    list2sub.append(feature)
            for x in list2sub:
                temp_list.remove(x)
            list2sub=[]
            str1=' '.join(temp_list)
            temp_list=[]
            self.new_x_train.append(str1)
        return self.new_x_train




    def random_select(self,selectk):
        import random 

        vectorizer=self.cv
        vectorizer.fit_transform(self.x_train)
        pool=vectorizer.get_feature_names()
        random.shuffle(pool)
        random_pool=pool[0:selectk]

        temp_list=[]
        self.new_x_train=[] #revised train set with only random terms
        list2sub=[] #temp list that holds elements to subtract

        for txt2 in self.x_train:
            temp_list=txt2.split()
            for feat in temp_list:
                if feat not in random_pool:
                    list2sub.append(feat)
            for x in list2sub:
                temp_list.remove(x)
            list2sub=[]
            str1=' '.join(temp_list)
            self.new_x_train.append(str1)
        return self.new_x_train


    def plot_confusion_matrix(self,cm, classes,normalize=False,title='Confusion matrix'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools
        cmap=self.plt.cm.Blues
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, self.np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        self.plt.imshow(cm, interpolation='nearest', cmap=cmap)
        self.plt.title(title)
        self.plt.colorbar()
        tick_marks = self.np.arange(len(classes))
        self.plt.xticks(tick_marks, classes, rotation=45)
        self.plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            self.plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        self.plt.ylabel('True label')
        self.plt.xlabel('Predicted label')
        self.plt.tight_layout()







    