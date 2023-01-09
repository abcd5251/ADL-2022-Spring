
# -*- coding:utf-8 -*-

import pandas as pd

import numpy as np
import os
from tqdm.notebook import tqdm


from utils import get_rouge
import json

from pathlib import Path
from random import random, seed
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
import os
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers.optimization import Adafactor
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 520,      
    'batch_size': 1,
    'learning_rate':0.0001,
    'n_epochs':15,
    'data_dir':  "./data/",   # Directory to the dataset
    'max_source_length': 1280,
    'padding': True,
    'pretrained_model' : "google/mt5-small",
    'save_path': './model.ckpt',  
    'early_stop': 200,  
    'valid_ratio': 0.05    
}

dataset_path = Path(config['data_dir'] + "train.jsonl")

article = []
title = []

with open(dataset_path, 'r', encoding = "utf-8") as f: # read data 
    for line in f.readlines():
        data = json.loads(line)
        
        title.append(data["title"])
        article.append(data["maintext"])

x_train, x_val, y_train, y_val = train_test_split( article , title , test_size=config["valid_ratio"], random_state=config["seed"])


test_path = Path(config['data_dir'] + "public.jsonl")

test_title = []
test_article = []
test_id = []

with open(test_path, 'r', encoding = "utf-8") as f: # read data 
    for line in f.readlines():
        data = json.loads(line)
        
        test_title.append(data["title"])
        test_article.append(data["maintext"])
        test_id.append(data["id"])
        
model =  MT5ForConditionalGeneration.from_pretrained(config["pretrained_model"]).to(device) # 
tokenizer = T5Tokenizer.from_pretrained(config["pretrained_model"])

padding = "max_length" if config["padding"] else False


class Data_Converter(Dataset):

    def __init__(self, x, datatype , y=None):
        
        self.type = datatype
        if y is None:
            self.y = y
        else:
            ignore_pad_token_for_loss = True
            with tokenizer.as_target_tokenizer():
                self.y = tokenizer(y, max_length = config["max_source_length"] ,padding=padding, truncation=True ,return_tensors="pt")
            if  self.type == "train":   
                if padding == "max_length" and ignore_pad_token_for_loss: # replace padding token id's of the labels by -100
                    self.y["input_ids"] = torch.tensor([[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in self.y["input_ids"]])
            
           
        
        self.x =  tokenizer(x, max_length = config["max_source_length"] ,padding=padding, truncation=True ,return_tensors="pt")  # tokenizer
        
    def __getitem__(self, idx):
        
        if self.y is None:
            
            return self.x["input_ids"][idx] , self.x["attention_mask"][idx]
        
        else:
        
            return self.x["input_ids"][idx] , self.x["attention_mask"][idx] , self.y["input_ids"][idx]

    def __len__(self):
        
        return len(self.x["input_ids"])


train_dataset = Data_Converter(x_train, "train" , y_train )

valid_dataset = Data_Converter(x_val, "valid", y_val )

test_dataset = Data_Converter(test_article , "test")

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)  

valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)  



def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def trainer(train_loader , model, config, device):

    print("start training")
    #optimizer = torch.optim.RAdam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    #lr_scheduler = AdafactorSchedule(optimizer)

    if not os.path.isdir('./models/'):
        os.mkdir('./models/') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x , x_a , y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, x_a , y = x.to(device), x_a.to(device), y.to(device)   # Move your data to device.
            
            loss = model(input_ids=x, attention_mask=x_a, labels=y).loss
            
            loss.backward()                    # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        
        mean_train_loss = sum(loss_record)/len(loss_record)
        torch.save(model.state_dict(), config['save_path']) # Save  best model
        
        
        model.eval() # Set your model to evaluation mode.
        refs = []
        val_pred = []
        for x, x_a , y in valid_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
        
                    output = model.generate(input_ids = x , max_length=128, num_beams = 6)
                    for i in range(len(output)) : 
                        out = tokenizer.decode(output[i], skip_special_tokens=True)
                        answer = tokenizer.decode(y[i], skip_special_tokens=True)
                        refs.append(answer)
                        val_pred.append(out)

        print(json.dumps(get_rouge(val_pred, refs), indent=2))

       
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}')


same_seed(config['seed'])
trainer(train_loader, model, config, device)



