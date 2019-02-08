class FeatureSelection():
    def __init__(self,target_cat,partition_by_id,):
        from sklearn.feature_extraction.text import CountVectorizer
        import numpy as np
        from scipy import stats as st
        from scipy import sparse as sp 
        import pandas as pd 
        import matplotlib.pyplot as plt


        self.sp=sp
        self.plt=plt
        self.pd=pd
        self.st=st
        self.cv=CountVectorizer(lowercase =False,binary=True)
        self.np=np
        self.partition_by_id=partition_by_id # must be integer 
        self.target_cat = target_cat #must be CCAT OR GCAT GFAS GWEA etc
        self.x_train=[]
        self.y_train=[]
        self.x_test=[]
        self.y_test=[]
        self.mask=[] # this is an 1-d array containing ones and zeros, for on is the rel_feature and zero the non rel, it depends of the rel_train vocabulary
        self.is_wknd_train=[]#this contains if the train document was written on a weekend 
        self.is_wknd_test=[] #this contains if the test document was written on a weekend 
        self.file_per_day_array=[]# an array that contains the number of relevant docs that day
        self.new_x_train=[]
        self.x_rel_train=[] #this contains the relevant train documents
        self.x_rel_train_pool=[] #this contains the feature of the relevant train documents
        self.list_2_zero=[] # this is a list that keeps all relevant terms that appear 5% or less in train set, these will be given a score 0,as a common start point for all feautereselection methods



    def split_data(self,data): #put this method under the open csv loop data must be in panda form
        self.data= data
        self.data['topic_bool']=self.data['topic'].map(lambda text :self.__target_category(text))
        filecounter=0 #this helps the uniform method to define the step interval of uniformity it can be a file or a day (maybe MONTH ???)
        for index,row in self.data.iterrows():
            if ( int(row['filename']) < self.partition_by_id ):
                text1=str(row['text'])+str(row['title'])
                self.x_train.append(text1)
                # self.is_wknd_train.append(row['is_wkdn'])
                self.y_train.append(row['topic_bool'])
                if(row['topic_bool']==1):
                    filecounter=filecounter+1
                    # self.x_rel_train.append(text1)
                    # temp_list=text1.split()
                    # temp_list=list(set(temp_list))
                    # for x in temp_list:
                    #     if x not in self.x_rel_train_pool:
                    #         self.x_rel_train_pool.append(x)
            else:
                self.x_test.append(str(row['text'])+str(row['title']))
                self.y_test.append(row['topic_bool'])
                # self.is_wknd_test.append(row['is_wkdn'])
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
        
        temp_list=[]
        self.new_x_test=[] #revised test set with only random terms 
        list2sub=[] #temp list that holds elements to subtract

        for txt2 in self.x_test:
            temp_list=txt2.split()
            for feat in temp_list:
                if feat not in random_pool:
                    list2sub.append(feat)
            for x in list2sub:
                temp_list.remove(x)
            list2sub=[]
            str1=' '.join(temp_list)
            self.new_x_test.append(str1)

        return self.new_x_train,self.new_x_test


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




    def transform_features(self,x_train,x_test,score,topk):

        # make a list from topk+1 to the end, in order to remove those columns from test and train
        # columns2_sub=[]
        # for x in final_pval[topk:None]:
        #     columns2_sub.append(x[1])


        # ########### TRANSFORM X_TRAIN ############
        mask = self.np.zeros(len(score), dtype=bool) ######## this code is from sklearns selectkbest it makes a mask with ones in selected features and zeros to not selected
        mask[self.np.argsort(score, kind="mergesort")[-topk:]] = True


        # arr=self.np.matrix(score) # transform the list into a matrix
        # arr=self.np.where(arr > threshold, 1, 0) # select only those features above threshold, make them 1 others zero

        # y = self.sp.spdiags(mask, 0, mask.size, mask.size) # this is the diagonal matrix containing only the selected indices as 1
        # y matrix is n_features * n_features with ones in selected indices and zeros otherwise
        
        # x_train=x_train * y  # [docs x n_features] * [ n_features x n_features] this gives the transformed x_train 
        x_train= x_train[:,mask]
        x_test = x_test[:,mask]


        # x_test=x_test * y  # [docs x n_features] * [ n_features x n_features] this gives the transformed x_test 


        # cols = columns2_sub
        # rows = []
        # mat=x_train
        # if len(rows) > 0 and len(cols) > 0:
        #     row_mask = self.np.ones(mat.shape[0], dtype=bool)
        #     row_mask[rows] = False
        #     col_mask = self.np.ones(mat.shape[1], dtype=bool)
        #     col_mask[cols] = False
        #     x_train= mat[row_mask][:,col_mask]
        # elif len(rows) > 0:
        #     mask = self.np.ones(mat.shape[0], dtype=bool)
        #     mask[rows] = False
        #     x_train= mat[mask]
        # elif len(cols) > 0:
        #     mask = self.np.ones(mat.shape[1], dtype=bool)
        #     mask[cols] = False
        #     x_train= mat[:,mask]
        # else:
        #     x_train= mat

        # ######## transform x_test ###############
        # cols = columns2_sub
        # rows = []
        # mat=x_test
        # if len(rows) > 0 and len(cols) > 0:
        #     row_mask = self.np.ones(mat.shape[0], dtype=bool)
        #     row_mask[rows] = False
        #     col_mask = self.np.ones(mat.shape[1], dtype=bool)
        #     col_mask[cols] = False
        #     x_test= mat[row_mask][:,col_mask]
        # elif len(rows) > 0:
        #     mask = self.np.ones(mat.shape[0], dtype=bool)
        #     mask[rows] = False
        #     x_test= mat[mask]
        # elif len(cols) > 0:
        #     mask = self.np.ones(mat.shape[1], dtype=bool)
        #     mask[cols] = False
        #     x_test= mat[:,mask]
        # else:
        #     x_test= mat


        return x_train,x_test


    def remove_non_rel_features(self,x_train,x_test):
        mask=self.mask
        x_train= x_train[:,mask]
        x_test = x_test[:,mask]

        return x_train,x_test





    def quick_uniform(self,x_rel_train):
        # in the beggining creates a list to subtract all non relevant documents
        # the result in a x_rel matrix containing all the features but only the rel documents
        x_rel_train=x_rel_train

        amount_of_documents=x_rel_train.shape[0] # this is the amount of only relevant documents
        amount_of_features=x_rel_train.shape[1]  # these are all the features

        #build the optimal uniform function
        opt_uni2=self.np.ones((amount_of_documents,1),dtype=int) 
        opt_uni2=opt_uni2.flatten() # from 2d make it one dimension
        k=0
        p_val=[]
        # list_2_zero=[] #make this list to actuppon the chi2 and mutual information
        while(k<amount_of_features):#check each feature what is its distribution for non_relevant put 0
            arr=x_rel_train[:,k]
            arr=arr.toarray()
            arr=self.np.where(arr > 0, 1, 0) # because we need just one, not the total amount of particular feature in that documnt so if there is above zero make it 1 (like binary countvectorizer) 
            # if (self.np.sum(arr)<=0.1*amount_of_documents/100):# cutoff threshold of relevant terms to avoid bad behaviour of ks2sample-uniform.
            #     p_val.append([0,k])
            #     list_2_zero.append(k)
            # else:
            #     arr=arr.flatten()
            #     p=self.st.ks_2samp(opt_uni2,arr)[1]
            #     p_val.append([p,k])
            arr=arr.flatten()
            p=self.st.ks_2samp(opt_uni2,arr)[1]
            p_val.append(p)
            # p_val.append([p,k])
            k=k+1


        # final_pval=sorted(p_val, key=lambda x: x[0],reverse =True)
        
        return p_val




    def quick_rdf(self,x_rel_train):
        # in the beggining creates a list to subtract all non relevant documents
        # the result in a x_rel matrix containing all the features but only the rel documents
        x_rel_train=x_rel_train
        amount_of_documents=x_rel_train.shape[0] # this is the amount of only relevant documents
        amount_of_features=x_rel_train.shape[1]  # these are all the features


        k=0
        score=[]
        while(k<amount_of_features):#check each feature what is its distribution for non_relevant put 0
            arr=x_rel_train[:,k]
            arr=arr.toarray()
            arr=self.np.where(arr > 0, 1, 0) # because we need just one, not the total amount of particular feature in that documnt so if there is above zero make it 1 (like binary countvectorizer)
            # if (self.np.sum(arr)<=0.1*amount_of_documents/100):# do this to have common start point with all methods
            #     score.append([0,k])
            # else:
            #     score.append([self.np.sum(arr),k]) 
            # score.append([self.np.sum(arr),k]) 
            score.append(self.np.sum(arr))
            k=k+1

        # final_score=sorted(score, key=lambda x: x[0],reverse =True)

        return score



    def get_x_rel_train(self,x_train,y_train):
        #the purpose of this function is to remove the non relevant documents and  get only the relevant corpus in a count_vectorizing fashion

        # y = self.sp.spdiags(y_train, 0, len(y_train), len(y_train)) #diagonal matrix containing the y_train

        # result= y * x_train # the result is the initial x_train matrix containing zero-rows according to the y_train

        # result = result[result.getnnz(1)>0]  # this code removes all zero-element- rows

        mask1 = self.np.array(y_train, dtype=bool) # this is an only rel_doc mask 

        x_rel_train= x_train[mask1] # this is a matrix containing only rel_docs in countvectorizer fashion

        mask = x_rel_train.getnnz(0)>0 # this is the mask, 1-d array containing only rel_features, it has the length of all features and ones in rel - zeros in nonrel

        x_rel_train= x_rel_train[:,mask] # this is the matrix with [rel_docs x rel_features]


        self.mask = mask

        # ind_list2_sub=[] 
        # i=0
        # while(i<len(y_train)):
        #     if(y_train[i]==0):
        #         ind_list2_sub.append(i)
        #     i=i+1
        
        # # this code removes the rows aka documents to create x_rel matrix
        # cols = []
        # rows = ind_list2_sub
        # mat=x_train
        # if len(rows) > 0 and len(cols) > 0:
        #     row_mask = self.np.ones(mat.shape[0], dtype=bool)
        #     row_mask[rows] = False
        #     col_mask = self.np.ones(mat.shape[1], dtype=bool)
        #     col_mask[cols] = False
        #     x_rel_train= mat[row_mask][:,col_mask]
        # elif len(rows) > 0:
        #     mask = self.np.ones(mat.shape[0], dtype=bool)
        #     mask[rows] = False
        #     x_rel_train= mat[mask]
        # elif len(cols) > 0:
        #     mask = self.np.ones(mat.shape[1], dtype=bool)
        #     mask[cols] = False
        #     x_rel_train= mat[:,mask]
        # else:
        #     x_rel_train= mat

        return x_rel_train



    def quick_uniform2(self,x_rel_train):
    # in the beggining creates a list to subtract all non relevant documents
    # the result in a x_rel matrix containing all the features but only the rel documents
        file_per_day_array=self.file_per_day_array
        x_rel_train=x_rel_train

        amount_of_features=x_rel_train.shape[1]  # these are all the features
        amount_of_documents=x_rel_train.shape[0]


        opt_uni2=self.np.ones((len(file_per_day_array),1),dtype=int).flatten()


        k=0
        score=[]
        day_score=[]
        doc_per_day=0
        position=0
        p_val=[]
        while(k<amount_of_features):#check each feature what is its distribution for non_relevant put 0
            arr=x_rel_train[:,k]
            arr=arr.toarray()
            arr=self.np.where(arr > 0, 1, 0) # because we need just one, not the total amount of particular feature in that documnt so if there is above zero make it 1 (like binary countvectorizer)
            while(doc_per_day<len(file_per_day_array)):
                temp_sum=self.np.sum(arr[position:position+file_per_day_array[doc_per_day]])
                for x in range(temp_sum):
                    day_score.append(doc_per_day)
                position=position+file_per_day_array[doc_per_day]
                doc_per_day=doc_per_day+1
            doc_per_day=0
            position=0
            # if (self.np.sum(day_score)<=0.1*amount_of_documents/100):# cutoff threshold of relevant terms to avoid bad behaviour of ks2sample-uniform.
            #     p_val.append([0,k])
            # else:  
            #     p=self.st.ks_2samp(list(opt_uni2),day_score)[1]
            #     p_val.append([p,k])
            p=self.st.ks_2samp(list(opt_uni2),day_score)[1]
            # p_val.append([p,k])
            p_val.append(p)
            day_score=[]
            k=k+1
        
        # final_pval=sorted(p_val, key=lambda x: x[0],reverse =True)     
        return p_val


    def feature_selection(self,x_rel_train):

        x_rel_train=x_rel_train
        amount_of_documents=x_rel_train.shape[0] # this is the amount of only relevant documents
        amount_of_features=x_rel_train.shape[1]  # these are all the features

        file_per_day_array=self.file_per_day_array

        opt_uni_order=self.np.arange(amount_of_documents)

        opt_uni_stamp=self.np.arange(len(file_per_day_array))



        k=0
        feat_sum=0
        score=[]
        day_score=[]
        doc_per_day=0
        position=0

        rdf_score=[]
        uni_order_score=[]
        uni_stamp_score=[]
    
        while(k<amount_of_features):#check each feature what is its distribution for non_relevant put 0
            arr=x_rel_train[:,k]
            arr=arr.toarray()
            arr=self.np.where(arr > 0, 1, 0) # because we need just one, not the total amount of particular feature in that documnt so if there is above zero make it 1 (like binary countvectorizer)
            feat_sum=self.np.sum(arr)

            ############## RDF ##############################
            rdf_score.append(feat_sum)
            ################ UNIFORM TIME STAMP ############################
            while(doc_per_day<len(file_per_day_array)):
                temp_sum=self.np.sum(arr[position:position+file_per_day_array[doc_per_day]])
                for x in range(temp_sum):
                    day_score.append(doc_per_day)
                position=position+file_per_day_array[doc_per_day]
                doc_per_day=doc_per_day+1
            doc_per_day=0
            position=0
            p_stamp=self.st.ks_2samp(opt_uni_stamp,day_score)[1]
            uni_stamp_score.append(p_stamp)
            day_score=[]

            ########### UNIFORM TIME ORDER ######################
            arr=self.np.nonzero(arr)[0]
            p_order=self.st.ks_2samp(opt_uni_order,arr)[1]
            uni_order_score.append(p_order)


            k=k+1

        return rdf_score, uni_order_score, uni_stamp_score 




        























        








    