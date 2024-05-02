import torch.nn as nn
import pandas as pd
import torch
import re
from dataset import Dataset
import random
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse
import wandb
import os
               
parser=argparse.ArgumentParser(description='StatNLP_termPJ')
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--epoch',type=int,default=20)
parser.add_argument('--lr',type=float,default=1e-7)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--non_en',type=bool,default=False)
args = parser.parse_args()

# to set the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    
set_seed(args.seed)
    
test_data=pd.read_csv("test.csv")
test_text=test_data['TEXT']
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model_name='roberta-base'

tokenizer=RobertaTokenizer.from_pretrained(model_name)
model=RobertaForSequenceClassification.from_pretrained(model_name,num_labels=3).to(device)

# load the trained model, you can access through the link written in the README.
MODEL_PATH="/home/labuser/Spring2024/Stat_NLP/Term_project/model/onlyenroberta_29_0.933_1e-07_32.pt"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
    
non_en_test=[]
for i,v_text in enumerate(test_text):
    
    # remove the "nan" data in test dataset
    if v_text!=v_text:
        non_en_test.append(i)
        continue
    # the texts shorter than 5 and non-english texts
    nonen_test=re.sub('[^a-zA-Z]','',v_text)
    if len(nonen_test)<5:
        non_en_test.append(i)
      
# if the non-english texts are not included for training
if args.non_en==False:
    test_text=[text for i,text in enumerate(test_text) if i not in non_en_test]
        
model.eval()

test_results=[]
nan_text=[]
for i,xs in enumerate(test_text):
    
    embedding=tokenizer.encode_plus(xs,add_special_tokens=True,max_length=128,padding='max_length', truncation=True,
                                          return_attention_mask=True,return_tensors='pt')
 
    ids=embedding['input_ids'].to(device)
    msk=embedding['attention_mask'].to(device)
    with torch.no_grad():
        test_outputs=model(input_ids=ids,attention_mask=msk)
        test_pred=test_outputs.logits.argmax(dim=1)
        
          

test_results.extend(test_pred.detach().cpu().numpy())

# if the non-english texts are not included for training, all of the texts are labeled as 0
if args.non_en==False:
    for i in non_en_test:
            test_results.insert(i,0)
        
print(len(test_results))
df=pd.DataFrame(columns=['ID','LABEL'])
df['ID']=test_data['ID']
df['LABEL']=test_results
df.to_csv("test_label_roberta.csv",index=False)
