# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 02:37:55 2018

@author: Ajaz
"""
#Importing Dependencies
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets,pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc,f1_score, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import cross_val_score
import pickle
from sklearn import svm
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

enron_files = ['enron1.tar.gz', 'enron2.tar.gz', 'enron3.tar.gz',
               'enron4.tar.gz', 'enron5.tar.gz', 'enron6.tar.gz']
dir_path="D:\\ML Assignment\\Compressed"
enron_folders=['enron1','enron2','enron3','enron4','enron5','enron6']
en_path=[]
#Extracting data sets 
def extract_file(path):
    #Creating enron data path
    for folders_path in enron_folders:
        en_path.append(dir_path+"\\"+folders_path)
    name=path.split('.')[0]
    #Checking if not already extracted by checking folder name
    if name not in en_path:
        tar=tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close
    else:
        print("{0} Already extracted ".format(path))
#looping through each compressed data set files
for files in enron_files:
    path=dir_path+"\\"+files
    #Calling extract file to extract compressed data set
    extract_file(path)

#Loading Dataset into data frame
def load_data(path):
    emails=datasets.load_files(path,categories=['ham','spam'],shuffle=True)        
    return pd.DataFrame({"Message":emails.data,"Label":emails.target})    

#Initializing data frame
df=pd.DataFrame({"Message":[],"Label":[]})
for folder in enron_folders:
    enron_path=dir_path+"\\"+folder
    dataframe=load_data(enron_path)
    df=df.append(dataframe)

# Dropping Duplicate rows from the data frame
df=df.drop_duplicates()
#Reset index of the rows
df=df.reset_index(drop=True)

#Removing punctuation from data set
df['Message']=df['Message'].astype(str).str.replace('[^\w\s]',' ')
print("Removed punctuation")
#Stemming the data sets
stemmer=SnowballStemmer("english")
df['Message']=df["Message"].apply(lambda x: [stemmer.stem(y) for y in x.split()])
print("Stemming complete")
#Removed digit
df['Message']=df['Message'].astype(str).str.replace('\d+',' ')
print("Digit removal complete")
#Remove line
df['Message']=df['Message'].astype(str).str.replace('\n',' ')
print("New line removal complete")
# Splitting data into training and test
X_train,X_test,y_train,y_test=train_test_split(df['Message'],df['Label'],test_size=0.3)

count_hams=df.groupby('Label').size()[0];
count_spams=df.groupby('Label').size()[1];
 
ham_training_set=0
spam_training_set=0
ham_test_set=0
spam_test_set=0

#Counting Spam and Ham in each set
#Counting in Training set
for label in y_train:
    if(label==0):
        ham_training_set=ham_training_set+1
    else:
        spam_training_set=spam_training_set+1        
print("\t  Ham \t Spam \n Training {0}  {1}".format(ham_training_set,spam_training_set))
#Counting in Test set
for label in y_test:
    if(label==0):
        ham_test_set=ham_test_set+1
    else:
        spam_test_set=spam_test_set+1
print("\t  Ham \t Spam \n Test     {0}  {1}".format(ham_test_set,spam_test_set))        
Data=('Training Ham','Training Spam','Test Ham','Test Spam')    
  
#Displaying count correspoding to each set of training and test records
records=[ham_training_set,spam_training_set,ham_test_set,spam_test_set]     
plt.bar(Data,records)

#Exploring data set
#Counting length of spam and ham messages in training set
print('Calculating length')
length_ham_list=[]
length_spam_list=[]
for message,label in zip(X_train,y_train):
    message_length=len(message)
    if(label==0):
        length_ham_list.append(message_length)
    else:
        length_spam_list.append(message_length)
#Adding length column to the data frame
df['length']=df['Message'].map(lambda text:len(text))
#Distribution of length of the spam emails
dsl=df.length[(df['Label'].abs()==1.0)]
#Distribution of length of the ham emails
dhl=df.length[(df['Label'].abs()==0.0)]
print('Distributed data')

df['log_length']=df['length'].map(lambda y:np.log(y))
df.boxplot(column='length',by='Label')
#Initializing Count vectorizer        
count_vect=CountVectorizer(encoding='latin1',stop_words='english')

#Bag of words
X_train_counts=count_vect.fit_transform(X_train) 

dataframes=pd.DataFrame(X_train_counts.toarray(),columns=count_vect.get_feature_names())
      
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)

X_train_tf = tf_transformer.fit_transform(X_train_counts)

feature_names=count_vect.get_feature_names()
print("Number of different words: {0}".format(len(feature_names)))

#classifier
clf=MultinomialNB().fit(X_train_counts,y_train)
print("Multinomial Classifier trained on the data")
#Cross Validation using KFold and evaluating scores
def build_pipeline():
    pipeline = Pipeline([
        ('count_vectorizer',   CountVectorizer(encoding='latin1',stop_words='english')),
        ('classifier',         MultinomialNB())])
    return pipeline
#Pipeline for Logistic Regression
def build_pipeline_LR():
    pipeline = Pipeline([
        ('count_vectorizer',   CountVectorizer(encoding='latin1',stop_words='english')),
        ('classifier',         LogisticRegression())])
    return pipeline
#Pipeline for SVC classifier
def build_pipeline_SVC():
    pipeline = Pipeline([
        ('count_vectorizer',   CountVectorizer(encoding='latin1',stop_words='english')),
        ('classifier',         svm.SVC(gamma='auto'))])
    return pipeline
#Splitting data set into 10 folds
kf = KFold(n_splits = 2)
#Confusion matrix to store records from Multinomial Naive Bayes
confusion = np.array([[0, 0], [0, 0]])#
#Confusion matrix to store records from Logistic Regression
confusion_LR = np.array([[0, 0], [0, 0]])
#Confusion matrix to store records from SVC
confusion_SVC = np.array([[0, 0], [0, 0]])
#Stores scores result from Multinomial NB
scores=[]
#Stores scores result from Logistic Regression
scores_LR=[]
#Stores scores result from SVC
scores_SVC=[]
#Building pipeline for each classifier
pipeline=build_pipeline()
pipelineLR=build_pipeline_LR()
pipelineSVC=build_pipeline_SVC()
#Looping through each set in X_train(data set) and storing results from each model
for train_index, test_index in kf.split(X_train):
      print("Train:", train_index, "Validation:",test_index)
      X_tr, X_tt = X_train.iloc[train_index], X_train.iloc[test_index] 
      y_tr, y_tt = y_train.iloc[train_index], y_train.iloc[test_index]
      
      pipeline.fit(X_tr, y_tr)
      pipelineLR.fit(X_tr,y_tr)
      pipelineSVC.fit(X_tr,y_tr)
      predictions = pipeline.predict(X_tt)
      confusion += confusion_matrix(y_tt, predictions)
      scores.append(f1_score(y_tt,predictions))
      
      predictionsLR = pipelineLR.predict(X_tt)
      confusion_LR += confusion_matrix(y_tt, predictionsLR)
      scores_LR.append(f1_score(y_tt,predictions))
      
      predictionsSVC = pipelineSVC.predict(X_tt)
      confusion_SVC += confusion_matrix(y_tt, predictionsSVC)
      scores_SVC.append(f1_score(y_tt,predictionsSVC))
      
#clfLR=LogisticRegression().fit(X_train_counts,y_train)
#clfSVC = svm.SVC(gamma='auto')

#clfSVC.fit(X_train_counts, y_train)
# save the Naive Bayes to disk
filename = 'finalized_model.sav'
pickle.dump(pipeline, open(filename, 'wb'))
#Ham word counts
ham_word_counts=clf.feature_count_[0,]
#Spam word counts
spam_word_counts=clf.feature_count_[1,]
tokens = pd.DataFrame({'token':count_vect.get_feature_names(), 'ham':ham_word_counts, 'spam':spam_word_counts}).set_index('token')
ham_list=tokens.sort_values(by='ham',ascending=False)
spam_list=tokens.sort_values(by='spam',ascending=False)

y_train_counts=count_vect.transform(X_test)

prediction=clf.predict(y_train_counts)
#prediction2=clfLR.predict(y_train_counts)
#prediction3=clfSVC.predict(y_train_counts)

count=0
#Model Evaluation
y_pred_proba = clf.predict_proba(y_train_counts)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc_curve = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc_curve))
plt.legend(loc=4)
plt.show()
#Training Error
predictions = clf.predict(X_train_counts)
mse = mean_squared_error(y_train, predictions)
rmse= np.sqrt(mse)
print('Training error')
print(rmse)
#validation error
mse = mean_squared_error(y_test, prediction)
rmse= np.sqrt(mse)
print('Test error')
print(rmse)
sc= cross_val_score(clf,X_train_counts , y_train, cv=10)
print("Cross Validation Score")
print(sc)
print('f1_score')
print(f1_score(y_test,prediction))
print('np mean')
print(np.mean(prediction==y_test))
print('accuracy score')
print(metrics.accuracy_score(y_test,prediction))
#calculate AUC
print('ROC AUC Score')
print(metrics.roc_auc_score(y_test,prediction))
#Classification Report
print(metrics.classification_report(y_test,prediction))
print(metrics.confusion_matrix(y_test, prediction))
print("Number of Ham:",count_hams,'\t',"Number of Spam:",count_spams)
print(ham_list['ham'][:20].plot.bar())
print(spam_list['spam'][:20].plot.bar())
# print message text for the false positives (ham incorrectly classified as spam)
i=0;
missed=[]
for x,y in zip(y_test,prediction):
    if(x>y):
       missed.append(df['Message'].iloc[i])
    i=i+1
vector=CountVectorizer()
tkn=vector.fit_transform(missed)
from operator import itemgetter
missed_spam=sorted(vector.vocabulary_.items(), key=itemgetter(1))