import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase





logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
"""
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )

    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to predict file.",
        default="./test.json"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to path to output predict file.",
    )
   

    args = parser.parse_args()
    return args
"""



@dataclass
class DataTrainingArguments:
   
    context_file: Optional[str] = field(
        default=None,
        metadata={"help": "An input of context_json."},
    )
    
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

def read_json(args):

    test = pd.DataFrame()
    


    context_dataset_path = Path(args.context_file)
    context_dataset = json.loads(context_dataset_path.read_text())
            
    
    # make test data

    dataset_path = Path(args.test_file)
    dataset = json.loads(dataset_path.read_text())
    
    id = []
    question = []
    paragraph1 = []
    paragraph2 = []
    paragraph3 = []
    paragraph4 = []
    labels = []
    

    for instance in dataset: # makes test data
             
        id.append(instance["id"])
        question.append(instance["question"])
        
        paragraph1.append(context_dataset[instance["paragraphs"][0]]) # these four as answer (sent2)
        paragraph2.append(context_dataset[instance["paragraphs"][1]])
        paragraph3.append(context_dataset[instance["paragraphs"][2]])
        paragraph4.append(context_dataset[instance["paragraphs"][3]])
        labels.append(3)
        
             
    test["id"] = id 
    test["question"] = question
    test["paragraph1"] = paragraph1
    test["paragraph2"] = paragraph2
    test["paragraph3"] = paragraph3
    test["paragraph4"] = paragraph4
    test["labels"] = labels
    

    return  test



@dataclass
class DataCollatorForMultipleChoice:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
   
   
   
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    test = read_json(data_args)

    
    data_files = {}
   
    test.to_csv("./test.csv",index = False , encoding = "utf-8")

    

    data_files["validation"] = "./test.csv"
    extension = data_files["validation"].split(".")[-1]
    raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None)
    
    
    model_name_or_path = "./context"

    config = AutoConfig.from_pretrained(
        model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name_or_path
    )

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Preprocessing the datasets.
    paragraph_names = ["paragraph1", "paragraph2", "paragraph3", "paragraph4"]
    def preprocess_function(examples):

        questions = [[context] * 4 for context in examples["question"]]

        four_sentences = [[examples[num][index] for num in paragraph_names] for index in range(len(examples["question"])) ] 
        
        questions= sum(questions, [])
        four_sentences = sum(four_sentences, [])
       
        tokenized_examples = tokenizer(
            questions,
            four_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if True:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)

        result = pd.DataFrame()
        result["id"] = test["id"]
        result["question"] = test["question"]
        ll = []
        for index , item in enumerate(preds):
            if preds[index] == 0:
                ll.append(test["paragraph1"][index])
            elif preds[index] == 1:
                ll.append(test["paragraph2"][index])
            elif preds[index] == 2:
                ll.append(test["paragraph3"][index])
            elif preds[index] == 3:
                ll.append(test["paragraph4"][index])
        result["context"] = ll
        result.to_csv("./result_context.csv" , encoding = "utf-8" , index = False)

        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset ,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    
    if True:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = dict(
        finetuned_from= model_name_or_path,
        tasks="multiple-choice",
        dataset_tags="test",
        dataset_args="regular",
        dataset="test",
        language="ch",
    )

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()