

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
import os
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers.optimization import Adafactor, AdafactorSchedule
import json
import argparse
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model =  MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device) # 
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
padding = "max_length" 


class Data_Converter(Dataset):  # tokenize and to pytorch format 

    def __init__(self, x, datatype , y=None):
        
        self.type = datatype
        if y is None:
            self.y = y
        else:
            ignore_pad_token_for_loss = True
            with tokenizer.as_target_tokenizer():
                self.y = tokenizer(y, max_length = 1280 ,padding=padding, truncation=True ,return_tensors="pt")
            if  self.type == "train":   
                if padding == "max_length" and ignore_pad_token_for_loss: # replace padding token id's of the labels by -100
                    self.y["input_ids"] = torch.tensor([[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in self.y["input_ids"]])
            
           
        
        self.x =  tokenizer(x, max_length = 1280 ,padding=padding, truncation=True ,return_tensors="pt")  # tokenizer
        
    def __getitem__(self, idx):
        
        if self.y is None:
            
            return self.x["input_ids"][idx] , self.x["attention_mask"][idx]
        
        else:
        
            return self.x["input_ids"][idx] , self.x["attention_mask"][idx] , self.y["input_ids"][idx]

    def __len__(self):
        
        return len(self.x["input_ids"])


def main(args):
    
    
    test_path = Path(args.input)

    #test_title = []
    test_article = []
    test_id = []

    with open(test_path, 'r', encoding = "utf-8") as f: # read data 
        for line in f.readlines():
            data = json.loads(line)
            
            #test_title.append(data["title"])
            test_article.append(data["maintext"])
            test_id.append(data["id"])
            
    

    model.eval()
    ckpt = torch.load("./model.ckpt", map_location=device)

    model.load_state_dict(ckpt)
    test_dataset = Data_Converter(test_article , "test")
    test_loader = DataLoader(test_dataset, batch_size= 1, shuffle=False, pin_memory=True)  

    def predict(test_loader, model, device):
        model.eval() # Set your model to evaluation mode.
        preds = []
        for x , x_a in tqdm(test_loader):
        
            x = x.to(device)
            
            with torch.no_grad(): 
                
                output = model.generate(input_ids = x  , max_length=128, num_beams = 6)
                for i in range(len(output)):
                    pred = tokenizer.decode(output[i], skip_special_tokens=True)
                    preds.append(pred)   
                
        return preds


    result_title = predict(test_loader , model , device)
        
     
    with open(args.output, 'w', newline='',encoding = "utf-8") as jsonfile: # save as json file

        for index , item in enumerate(result_title) :
            temp = {}
            temp["title"] = item
            temp["id"] = test_id[index]
            jsonfile.write(str(temp) + "\n")
    

    
         



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '--input')
    parser.add_argument('-output', '--output')
    args = parser.parse_args()
    main(args)
