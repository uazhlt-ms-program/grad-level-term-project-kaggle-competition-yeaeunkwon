import torch.nn as nn
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    
    def __init__(self,text,label,tokenizer):
        super().__init__()
        self.x=text
        self.y=label
        #self.x=text.tolist()
        #self.y=label.values
        self.tokenizer=tokenizer
        
    def __len__(self):
        
        return len(self.y)
    
    def __getitem__(self,idx):
        
        xs=self.x[idx]
        ys=self.y[idx]
        embedding=self.tokenizer.encode_plus(xs,add_special_tokens=True,max_length=128,padding='max_length', truncation=True,
                                          return_attention_mask=True,return_tensors='pt')
        
        return {'input_id':embedding['input_ids'].flatten(),
                'attention_mask':embedding['attention_mask'].flatten(),
                'label':torch.tensor(ys,dtype=torch.long)}
        
        