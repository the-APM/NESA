import json
import re
import numpy as np
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from random import shuffle
from sklearn.naive_bayes import MultinomialNB

with open('SAIL_HN-EN_train_codemixed.json') as data_file:    
    data_train = json.load(data_file)

##with open('HI-EN_Test.json') as data_file:    
##    data_test = json.load(data_file) 18461

#pprint(data) ##<-uncomment this to print the entire data set

xd=[]
y=[]

data_train[11270]['text']='ok'
data_train[11270]['sentiment']=0

e="a"
c=0 ##increase this at the end of the iteration
while (e!="0"):
    e=raw_input("Enter text for analysis ya press 0 to exit ")
    if(e!="0"):
        xd=[]
        y=[]
        for x in range(0,12937+c):
            if(x<12936):
                xd.insert(x,data_train[x]['text'])
                y.insert(x,data_train[x]['sentiment'])
            else:
                xd.insert(x,e)


        for i in range(0,12937+c):
            xd[i]=re.sub(r'http/[\w_-]+','url',xd[i])
            xd[i]=re.sub(r'http//[\w_-]+','url',xd[i])
            xd[i]=re.sub(r'http:/[\w_-]+','url',xd[i])
            xd[i]=re.sub(r'http://[\w_-]+','url',xd[i])
            xd[i]=re.sub(r'@[\w_-]+','someuser',xd[i])
            xd[i]=re.sub(r'#[\w_-]+','hashtag',xd[i])
            
            xd[i]=xd[i].replace(':-)','happy')
            xd[i]=xd[i].replace(':-(','sad')
            xd[i]=xd[i].replace(':-p','fun')
            xd[i]=xd[i].replace(':-d','troll')
            xd[i]=xd[i].replace('<3','love')
            xd[i]=xd[i].replace(':)','happy')
            xd[i]=xd[i].replace(':(','sad')
            xd[i]=xd[i].replace(':p','fun')
            xd[i]=xd[i].replace(':d','troll')
            xd[i]=xd[i].replace(';-p','happy')
            xd[i]=xd[i].replace(';p','happy')
            xd[i]=xd[i].replace(':-D','happy')
            xd[i]=xd[i].replace(':D','happy')
            xd[i]=xd[i].replace(":'-(","sad")
            xd[i]=xd[i].replace(":'(","sad")
            xd[i]=xd[i].replace(":'{","sad")

        print("Pre-processing done...")
        vectorizer = CountVectorizer()
        vectorizer                     
        CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                dtype='numpy.int64', encoding='utf-8', input='content',
                lowercase=True, max_df=1.0, max_features=None, min_df=1,
                ngram_range=(1, 1), preprocessor=None, stop_words={'english'},
                strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None, vocabulary=None)

        x_vec1=vectorizer.fit_transform(xd)

        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),stop_words={'english'},token_pattern=r'\b\w+\b', min_df=1)
        x_vec2= bigram_vectorizer.fit_transform(xd)
        
        print("Vectorization done...")
        xtraining_vec1=x_vec1[0:12936+c]
        xtest_vec1=x_vec1[12936+c:]
        xtraining_vec2=x_vec2[0:12936+c]
        xtest_vec2=x_vec2[12936+c:]
        ytraining_vec=y[0:12936+c]
        ytest_vec=y[12936+c:]

        u_clf = svm.SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        u_clf.fit(xtraining_vec1,y)

        b_clf = svm.SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        b_clf.fit(xtraining_vec2,y)

        nbu_clf = MultinomialNB().fit(xtraining_vec1,y)
        nbb_clf = MultinomialNB().fit(xtraining_vec2,y)

        print("Data-fitting done...")
        yu_res=u_clf.predict(xtest_vec1)
        yb_res=b_clf.predict(xtest_vec2)
        nb_res_yu=nbu_clf.predict(xtest_vec1)
        nb_res_yb=nbb_clf.predict(xtest_vec2)

        print("Predictions done...")
        y_res=nb_res_yb
        for a in range (0,len(yu_res)-1):
            if ((yu_res[a]==yb_res[a])&(yb_res[a]==nb_res_yu[a])&(nb_res_yu[a]!=nb_res_yb[a])):
                y_res[a]=yu_res[a]

        print y_res
        f=raw_input("Is the predicted sentiment correct?->Y/N=")
        if(f=='Y'):
            y.insert(x,y_res[0])
            
        else:
            r=input("Please enter the correct sentiment?->1/0/-1=")
            y.insert(x,r)
        c=c+1


