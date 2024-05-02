from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from tfidf_error_analysis import error_analysis
from nltk.stem.porter import *
from clean_tfidf import review_classifier
import argparse


parser=argparse.ArgumentParser(description='Review_classifier')
parser.add_argument('--non_en',type=bool,default=False)
parser.add_argument('--preprocessing',type=bool,default=None)
parser.add_argument('--ngram',type=tuple,default=(1,1))

args = parser.parse_args()


train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")
    
train_text=train_data['TEXT']
test_text=test_data['TEXT']
    
# drop the null data
train_data=train_data.dropna(axis=0)
    
train_text=train_data['TEXT']
test_text=test_data['TEXT']
    
train_label=train_data['LABEL']
    
non_en_train=[]
non_en_test=[]
    
#search the index of non-en texts and the texts shorter than 5
for i,t_text in enumerate(train_text):
    nonen_train=re.sub('[^a-zA-Z]','',t_text)
    if len(nonen_train)<5:
        non_en_train.append(i)
        
for i,text in enumerate(test_text):
    
    #"nan"text
    if text!=text:
        non_en_test.append(i)
        continue
    
    nonen_test=re.sub('[^a-zA-Z]','',text)
    
    if len(nonen_test)<5:
        non_en_test.append(i)
        
# if the non-english texts are not included for training 
if args.non_en==False:
        
    train_text=[text for i,text in enumerate(train_text) if i not in non_en_train]
    train_label=[label for i,label in enumerate(train_label) if i not in non_en_train]
    test_text=[text for i,text in enumerate(test_text) if i not in non_en_test]
     
        
model=review_classifier(preprocessing=args.preprocessing,n_gram=args.ngram)
tfidf_train=model.encoder_train(train_text,train_label)
print(tfidf_train.shape)
tfidf_valid=model.encoder_predict(test_text)
model.classifier_train(tfidf_train,train_label)
pred=model.classifier_predict(tfidf_valid)
pred=list(pred)

# if the non-english texts are not included for training, all of the texts are labeled as 0
if args.non_en==False:
    for i in non_en_test:
        pred.insert(i,0)

# save the result to a csv file
df=pd.DataFrame(columns=['ID','LABEL'])
df['ID']=test_data['ID']
df['LABEL']=pred
df.to_csv("test_label_test.csv",index=False)
