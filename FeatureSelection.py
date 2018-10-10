class FeatureSelection():
    def __init__(self,target_cat,partition_by_id):
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
        self.file_per_day_array=[]# an array that contains the number of rel_files that day


    def split_data(self,data): #put this method under the open csv loop data must be in panda form
        self.data= data
        self.data['topic_bool']=self.data['topic'].map(lambda text :self.__target_category(text))
        filecounter=0 #this helps the uniform method to define the step interval of uniformity it can be a file or a day (maybe MONTH ???)
        for index,row in self.data.iterrows():
            if ( int(row['filename']) < self.partition_by_id ):
                self.x_train.append(row['text'])
                self.y_train.append(row['topic_bool'])
                filecounter=filecounter +1 
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


    def uniform(self,step_interval):
        self.step_interval=step_interval #step interval can be either 'single' or day
        x_rel_train=[]
        for txt,is_rel in zip(self.x_train,self.y_train): #take only relative
            if (is_rel==1):
                x_rel_train.append(txt)


        rel_vectorizer=self.cv
        rel_vectorizer.fit_transform(x_rel_train)
        rel_pool=rel_vectorizer.get_feature_names() #this is the relative features pool in alphabetic order

        if (self.step_interval=='single'):
            timeline=sum(self.y_train) # the sum of ralative documents
            step=1/timeline

            #first build the optimal cumulative uniform discrete function
            cumulative=[]
            #axes=[]
            k=1 #counter
            while(k<=timeline):
                cumulative.append(step*k)
                #axes.append(k)
                k=k+1

            #calculate the cumulative distribution for every feature and get the p value
            p_val=[]
            k=0
            relative_k=1 #counter of the step interval 
            temp_sum=0
            temp_cumulative=[]
            for feat in rel_pool:
                while(k<timeline):
                    if feat not in x_rel_train[k].split():
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
            self.p_val_feat = self.pd.DataFrame(datainput=d)
        else: #this means that the interval will be a day
            timeline=len(self.file_per_day_array)
            step=1/timeline
            #first build the optimal cumulative uniform discrete function
            cumulative=[]
            #axes=[]
            k=1 #counter
            while(k<=timeline):
                cumulative.append(step*k)
                #axes.append(k)
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
                        if feat in self.x_rel_train[file].split():
                            temp_sum=temp_sum+1
                        rel_file=rel_file +1
                        file=file+1
                    if (temp_sum!=0):
                        temp_cumulative.append(temp_sum/self.file_per_day_array[k])
                    temp_sum=0
                    rel_file=0#set this to zero to count the next num of files_per day
                    k=k+1
                tc=self.np.array(temp_cumulative)/(sum(temp_cumulative)) #subtract 1 because the last iteration of relative_k would be faulty
                while(tc_count<len(tc)-1):
                    tc[tc_count+1]=tc[tc_count]+tc[tc_count+1]
                    tc_count=tc_count+1
                cc=self.np.array(cumulative)
                p=self.st.ks_2samp(cc, tc)[1]
                p_val.append(p)
                temp_cumulative=[]
                temp_sum=0
                k=0
                tc_count=0
            d = {'p_val': p_val, 'feat': rel_pool}
            self.p_val_feat = pd.DataFrame(data=d)


