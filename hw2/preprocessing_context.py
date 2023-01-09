import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging

config = {
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

        dataset_path = Path( config['data_dir'] + f"{split}.json")
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        id = []
        question = []
        paragraph1 = []
        paragraph2 = []
        paragraph3 = []
        paragraph4 = []
        label_paragraph = []
        start_pos = []
        text = []


        for instance in dataset:  # makes train data
             
                       
             id.append(instance["id"])
             question.append(instance["question"]) # use this as question (sent1)
             
             paragraph1.append(context_dataset[instance["paragraphs"][0]]) # these four as answer (sent2)
             paragraph2.append(context_dataset[instance["paragraphs"][1]])
             paragraph3.append(context_dataset[instance["paragraphs"][2]])
             paragraph4.append(context_dataset[instance["paragraphs"][3]])
             label_paragraph.append(instance["paragraphs"].index(instance["relevant"]))
             start_pos.append(instance["answer"]["start"])
             text.append(instance["answer"]["text"])
          
        

        if split == "train":
           
            train["id"] = id
            train["question"] = question
            train["paragraph1"] = paragraph1
            train["paragraph2"] = paragraph2
            train["paragraph3"] = paragraph3
            train["paragraph4"] = paragraph4
            train["label"] = label_paragraph
            train["start_pos"] = start_pos
            train["text"] = text

        elif split == "valid" :

            valid["id"] = id
            valid["question"] = question
            valid["paragraph1"] = paragraph1
            valid["paragraph2"] = paragraph2
            valid["paragraph3"] = paragraph3
            valid["paragraph4"] = paragraph4
            valid["label"] = label_paragraph
            valid["start_pos"] = start_pos
            valid["text"] = text

    
    # make test data

    dataset_path = Path( config['data_dir'] + "test.json")
    dataset = json.loads(dataset_path.read_text())
    logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")
    id = []
    question = []
    paragraph1 = []
    paragraph2 = []
    paragraph3 = []
    paragraph4 = []
    label_paragraph = []
    start_pos = []
    text = []
    

    for instance in dataset: # makes test data
             
        id.append(instance["id"])
        question.append(instance["question"])
        
        paragraph1.append(context_dataset[instance["paragraphs"][0]]) # these four as answer (sent2)
        paragraph2.append(context_dataset[instance["paragraphs"][1]])
        paragraph3.append(context_dataset[instance["paragraphs"][2]])
        paragraph4.append(context_dataset[instance["paragraphs"][3]])
        
             
    test["id"] = id 
    test["question"] = question
    test["paragraph1"] = paragraph1
    test["paragraph2"] = paragraph2
    test["paragraph3"] = paragraph3
    test["paragraph4"] = paragraph4
    

    return train , valid , test

train, valid , test = read_json()

train.to_csv("./train.csv",index = False ,encoding = "utf-8-sig")
valid.to_csv("./valid.csv",index = False ,encoding = "utf-8-sig")
test.to_csv("./test.csv",index = False ,encoding = "utf-8-sig")