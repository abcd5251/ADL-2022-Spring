import json
import csv
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import default_data_collator, Trainer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_callback import EarlyStoppingCallback
import numpy as np
from datasets import load_metric, load_dataset
import argparse


tokenizer = AutoTokenizer.from_pretrained('./runs/finetune_out/checkpoint-2560')
model = T5ForConditionalGeneration.from_pretrained('./runs/finetune_out/checkpoint-2560')
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
tokenizer.pad_token = tokenizer.eos_token
metric = load_metric("sacrebleu")
max_input_length = 60
max_target_length = 30

def preprocess_function(examples):
    inputs = [ex for ex in examples['inputs']]
    targets = [ex for ex in examples['target']]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding='max_length',
        add_special_tokens=True,
    )

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True, padding='max_length',
            add_special_tokens=True,
        )

    model_inputs["labels"] = labels["input_ids"]
   
    return model_inputs


class OTTersDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data):
        tokenizer.pad_token = tokenizer.eos_token
        self.encodings = tokenizer(data['inputs'], padding=True, truncation=True) 
        with tokenizer.as_target_tokenizer():
            self.targets = tokenizer(data['targets'], padding=True, truncation=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.targets['input_ids'][idx]
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def read_data(data_dir):

    splits = ['train', 'dev', 'test']
    datasets = {}
    for split in splits:
        directory = os.path.join(data_dir, split)
        datasets[split] = load_dataset(directory, data_files=['text.csv'])
        if split != 'test':
            datasets[split] = datasets[split].map(
                preprocess_function,
                batched=True,
                remove_columns=['inputs', 'target'],
            )['train']
        else:
            datasets[split] = datasets[split]['train']
    return datasets['train'], datasets['dev'], datasets['test']

def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        print(preds[0])
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

if __name__ == "__main__":

    print('reading dataset')
    dataset_dir = os.path.join('./data', 'out_of_domain')
    train_dataset, eval_dataset, test_dataset = read_data(dataset_dir) 



 # test
    model.to('cpu')
    inputs = tokenizer(test_dataset['inputs'], return_tensors="pt", padding=True)
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    #print(output_sequences)
    predictions = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    predictions = [pred.strip() for pred in predictions]
    #print(len(predictions))
    #print(len(test_dataset['target']))

    #targets = []
    #for item in test_dataset['target']:
        
    #    targets.append(item.split(" "))
    #print(len(targets))
    #print(len(predictions))
    #print(test_dataset['target'])
    output_prediction_file = os.path.join("./", "out_predictions.txt")
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        writer.write("\n".join(predictions))
    
    result = metric.compute(predictions= [predictions], references= [test_dataset['target']])
    result = {"bleu": result["score"]}
    print(result)

    #trainer.save_model(args.output_dir)