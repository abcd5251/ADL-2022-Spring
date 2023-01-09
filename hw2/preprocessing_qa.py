import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging

config = {
    'seed': 370,      
    'batch_size': 3,
    'model_checkpoint': "bert-base-chinese",
    'data_dir':  "./dataset/"  # Directory to the dataset 
}

def read_json():

    train = pd.DataFrame()
    valid = pd.DataFrame()
    test = pd.DataFrame()
    
    context_dataset_path = Path(config['data_dir'] + "context.json")
    context_dataset = json.loads(context_dataset_path.read_text())
    

    for split in ["train", "valid"]:
        
        id = []
        question = []
        context = []
        answers = []

        dataset_path = Path( config['data_dir'] + f"{split}.json")
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")
        
        for instance in dataset:  # makes train and valid data
                    
             id.append(instance["id"])
             question.append(instance["question"]) 
             context.append(context_dataset[instance["relevant"]])
             temp_answer = {}
             temp_answer["text"] = instance["answer"]["text"]
             temp_answer["answer_start"] = instance["answer"]["start"]
            
             answers.append(dict(temp_answer))

        if split == "train":
           
            train["id"] = id
            train["question"] = question
            train["context"] = context
            train["answers"] = answers

        elif split == "valid" :
            valid["id"] = id
            valid["question"] = question
            valid["context"] = context
            valid["answers"] = answers

    
    # make test data

    dataset_path = Path( config['data_dir'] + "test.json")
    dataset = json.loads(dataset_path.read_text())
    logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")
    id = []
    question = []
    context = []
    

    for instance in dataset: # makes test data
             
        id.append(instance["id"])
        question.append(instance["question"]) 
        
             
    test["id"] = id 
    test["question"] = question
    

    return train , valid , test

train, valid , test = read_json()

train.to_csv("./train.csv",index = False ,encoding = "utf-8-sig")
valid.to_csv("./valid.csv",index = False ,encoding = "utf-8-sig")
test.to_csv("./test.csv",index = False ,encoding = "utf-8-sig")