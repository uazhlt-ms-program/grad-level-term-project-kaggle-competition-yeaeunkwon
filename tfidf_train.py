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
import argparse


parser=argparse.ArgumentParser(description='Review_classifier')
parser.add_argument('--non_en',type=bool,default=False)
parser.add_argument('--preprocessing',type=bool,default=None)
parser.add_argument('--ngram',type=tuple,default=(1,1))

args = parser.parse_args()

class review_classifier():
    
    def __init__(self,preprocessing,n_gram):
        super().__init__()
        
        if preprocessing:
            my_processor=self.my_processor
        else:
            my_processor=preprocessing
        self.encoder=TfidfVectorizer(preprocessor=my_processor,ngram_range=n_gram)
        self.model=LogisticRegression(random_state=42, max_iter=100,solver='lbfgs',C=5,class_weight='balanced')
        self.lemmatizer=WordNetLemmatizer()
        self.stemmer=PorterStemmer()
        
    def my_processor(self,text):
        new_d=re.sub('<br />',' ',text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in word_tokenize(new_d)]
        stemmed_words=[self.stemmer.stem(word) for word in lemmatized_words]
        new_d=re.sub('read','[tag]',' '.join(stemmed_words))
        new_d=re.sub('book','[tag]',new_d)
        
        return new_d
    
    def encoder_train(self,data,label):
        
        return self.encoder.fit_transform(data,label)
    
    def encoder_predict(self,data):
        
        return self.encoder.transform(data)
    
    def classifier_train(self,data,label):
        
        self.model.fit(data,label)
        
    def classifier_predict(self,data):
        
        pred=self.model.predict(data)
        return pred
        
   
        
        
    
    
if __name__=="__main__":
    
    train_data=pd.read_csv("train.csv")
    
    train_text=train_data['TEXT']
    
    # drop the null data
    train_data=train_data.dropna(axis=0)
    
    train_text=train_data['TEXT']
    
    train_label=train_data['LABEL']
    train_text,valid_text,train_label,valid_label=train_test_split(train_text,train_label,test_size=0.3,random_state=42)
    
    non_en_train=[]
    non_en_valid=[]
    
    #search the index of non-en texts and the thexts shorter than 5
    for i,t_text in enumerate(train_text):
        nonen_train=re.sub('[^a-zA-Z]','',t_text)
        if len(nonen_train)<5:
            non_en_train.append(i)
        
    for i,v_text in enumerate(valid_text):
        nonen_valid=re.sub('[^a-zA-Z]','',v_text)
        if len(nonen_valid)<5:
            non_en_valid.append(i)

    # if the non-english texts are not included for training
    if args.non_en==False:
        
        train_text=[text for i,text in enumerate(train_text) if i not in non_en_train]
        valid_text=[text for i,text in enumerate(valid_text) if i not in non_en_valid]
        train_label=[label for i,label in enumerate(train_label) if i not in non_en_train]
       
        
    model=review_classifier(preprocessing=args.preprocessing,n_gram=args.ngram)
    tfidf_train=model.encoder_train(train_text,train_label)
    print(tfidf_train.shape)
    tfidf_valid=model.encoder_predict(valid_text)
    model.classifier_train(tfidf_train,train_label)
    pred=model.classifier_predict(tfidf_valid)
    pred=list(pred)

    # if the non-english texts are not included for training, all of the texts are labeled as 0
    if args.non_en==False:
        for i in non_en_valid:
            pred.insert(i,0)
        
    print(f1_score(valid_label,pred,average="macro",zero_division=0))
