class FeatureSelection():
    def __init__(self,target_cat,partition_by_id,):
        from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
        import numpy as np
        from scipy import stats as st
        import pandas as pd 
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
        # import matplotlib.pyplot as plt
        # self.plt=plt


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


    def uniform(self,step_interval,decision_thres=0.5):
        self.step_interval=step_interval #step interval can be either 'single' or day
        self.x_rel_train=[]
        self.decision_thres=decision_thres #it can be from 1 to 0.001 --? 1 means even one occurence count as uniform
        for txt,is_rel in zip(self.x_train,self.y_train): #take only relative
            if (is_rel==1):
                self.x_rel_train.append(txt)


        rel_vectorizer=self.cv
        rel_vectorizer.fit_transform(self.x_rel_train)
        rel_pool=rel_vectorizer.get_feature_names() #this is the relative features pool in alphabetic order
        print(len(rel_pool))
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

            d = {'p_val': p_val, 'feat': rel_pool}
            self.p_val_feat = self.pd.DataFrame(data=d)
        else: #this means that the interval will be a day
            # fig, ax = self.plt.subplots(1, 1)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
            d = {'p_val': p_val}
            self.p_val_feat = self.pd.DataFrame(data=d) #this is a dataframe containing the values and the features 
            # ax.step(cc, 'ro')
            # ax.step(tc,'bo')
            # self.plt.show()

    def rdf(self,min_df):
        from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
        self.min_df=min_df
        self.x_rel_train=[]
        for txt,is_rel in zip(self.x_train,self.y_train): #take only relative
            if (is_rel==1):
                self.x_rel_train.append(txt)

        rel_vectorizer=CountVectorizer(lowercase =False)
        rel_vectorizer.fit_transform(self.x_rel_train)
        rel_pool=rel_vectorizer.get_feature_names() #this is the relative features pool in alphabetic order



# print ("the pool of relevant terms has  " + str(len(rel_pool)) +" features.")
# subtrack from x_train all other features that are not included with regards to rel_pool
        temp_list=[]
        new_x_train=[] #revised train set with only rel terms 
        list2sub=[] #temp list that holds elements to subtract

        for txt2 in self.x_train:
            temp_list=txt2.split()
            for feat in temp_list:
                if feat not in rel_pool:
                    list2sub.append(feat)
            for x in list2sub:
                temp_list.remove(x)
            list2sub=[]
            str1=' '.join(temp_list)
            new_x_train.append(str1)
            
        count_vectorizer=CountVectorizer(lowercase =False,min_df=self.min_df)#documet frequency of a rel_term threshold 
        count_vectorizer.fit_transform(new_x_train)
        self.rdf_rel_pool=count_vectorizer.get_feature_names()





