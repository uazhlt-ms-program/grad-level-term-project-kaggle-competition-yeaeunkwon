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
parser.add_argument('--epoch',type=int,default=30)
parser.add_argument('--lr',type=float,default=1e-7)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--non_en',type=bool,default=False)
args = parser.parse_args()

wandb.init(project="StatNLP")
wandb.run.name=f"onlyenRoberta_{args.lr}_{args.batch_size}"
wandb.config.update(args)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def train_fn(train_dataloader,device,model,optim):
    
    lossK=0.0
    train_preds=[]
    train_true=[]
    model.train()
    for i,xs in enumerate(train_dataloader):
        ids=xs['input_id'].squeeze(1).to(device)
        xmsk=xs['attention_mask'].squeeze(1).to(device)
        ys=xs['label'].to(device)
        outputs=model(ids,xmsk,labels=ys)
    
        pred=outputs.logits.argmax(dim=1)
        loss=outputs[0]
        
        lossK+=loss.item()
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        if i%100==0:
            print(i,lossK)
        
        train_preds.extend(pred.detach().cpu().numpy())
        train_true.extend(ys.detach().cpu().numpy())
        
    train_loss=lossK/len(train_dataloader)
    train_f1=f1_score(train_true, train_preds,average="macro",zero_division=0)
    
    return train_loss,train_f1

def prediction(valid_dataloader,device,model,true_label,non_en_valid):
    model.eval()
    valid_results=[]
    for i,xs in enumerate(valid_dataloader):
        ids=xs['input_id'].squeeze(1).to(device)
        msk=xs['attention_mask'].squeeze(1).to(device)
        with torch.no_grad():
            valid_outputs=model(input_ids=ids,attention_mask=msk)
            valid_pred=valid_outputs.logits.argmax(dim=1)
          

        valid_results.extend(valid_pred.detach().cpu().numpy())
        
    for i in non_en_valid:
        valid_results.insert(i,0)
    valid_f1=f1_score(true_label, valid_results,average="macro",zero_division=0)
    
    return valid_f1

def experiment_fn(train_datalodaer,valid_dataloader,device,model,valid_true_label,non_en_valid):
    
    best_f1=0.0
    optim=AdamW(model.parameters(),lr=args.lr,eps=1e-8)
    criterion=nn.CrossEntropyLoss()
    
    for ep in range(args.epoch):
        
        train_loss,train_f1=train_fn(train_dataloader,device,model,optim)
        print(f"train_loss : {train_loss} | train_f1 : { train_f1}")
        
        valid_f1=prediction(valid_dataloader,device,model,valid_true_label,non_en_valid)
        
        wandb.log({"train_loss":train_loss,"train_f1":train_f1,"valid_f1":valid_f1},step=ep)
    
        save_dict={"model_state_dict":model.state_dict(),
                   "optimizer_state_dict":optim.state_dict()}
    
        if best_f1==0 or best_f1<valid_f1:
            best_f1=valid_f1
            model_name=f"roberta_{ep}_{valid_f1:.3f}_{args.lr}_{args.batch_size}.pt"
            model_path="/home/labuser/Spring2024/Stat_NLP/Term_project/model"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(save_dict,os.path.join(model_path,model_name))
            
        print(f"valid_f1 : {valid_f1} | best f1 : {best_f1}")
    
        
if __name__=="__main__":
    
    
    set_seed(args.seed)
    
    train_data=pd.read_csv("train.csv")
    test_data=pd.read_csv("test.csv")
    
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name='roberta-base'

    tokenizer=RobertaTokenizer.from_pretrained(model_name)
    model=RobertaForSequenceClassification.from_pretrained(model_name,num_labels=3).to(device)

    #remove null values
    train_data=train_data.dropna(axis=0)
    
    train_text=train_data['TEXT']
    test_text=test_data['TEXT']
    
    train_label=train_data['LABEL']
    train_text,valid_text,train_label,valid_label=train_test_split(train_text,train_label,test_size=0.3,random_state=42)
    
    non_en_train=[]
    non_en_valid=[]

    #search the index of non-en texts
    for i,t_text in enumerate(train_text):
        nonen_train=re.sub('[^a-zA-Z]','',t_text)
        if len(nonen_train)<5:
            non_en_train.append(i)
     
        
    for i,v_text in enumerate(valid_text):
        nonen_valid=re.sub('[^a-zA-Z]','',v_text)
        if len(nonen_valid)<5:
            non_en_valid.append(i)
            
    valid_true_label=valid_label   
          
    if args.non_en==False:
        
          
        train_text=[text for i,text in enumerate(train_text) if i not in non_en_train]
        valid_text=[text for i,text in enumerate(valid_text) if i not in non_en_valid]
        train_label=[label for i,label in enumerate(train_label) if i not in non_en_train]
        valid_label=[label for i,label in enumerate(valid_label) if i not in non_en_valid]
        
        

    train_dataset=Dataset(train_text,train_label,tokenizer)
    valid_dataset=Dataset(valid_text,valid_label,tokenizer)

    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    valid_dataloader=DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=False)
    
    
    experiment_fn(train_dataloader,valid_dataloader,device,model,valid_true_label,non_en_valid)