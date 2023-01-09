import argparse
import json
import random
import torch
import numpy as np
'''
add  for T5 model
'''


import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)

import json
from collections import defaultdict
import spacy
from tqdm import tqdm
import csv

def checkhit(dialog,keywords):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    statistics = []
    hit_counts = {key: defaultdict(int) for key in keywords.keys()}
    hit_num = 0
    for index in range(2, len(dialog), 2):
        lemma_utterance = [token.lemma_ for token in nlp(dialog[index])]
        service_hits = defaultdict(int)
        for key, (one, multi) in keywords.items():
            #print(one)
            intersection = set(one) & set(lemma_utterance)
            '''
            one 代表只有一個關鍵字的key word 直接找就好了
            multi 代表是一個>一個字的詞,需要讓Utternace重組才能比較
            '''
            # check whether the word, the length is bigger than 2, is in the utterance
            for m in multi:
                unsplit_utterance = " ".join(lemma_utterance)
                if m in unsplit_utterance:
                    intersection.add(m)
            service_hits[key] += len(intersection)
            statistics += list(intersection)
            for hit in intersection:
                hit_counts[key][hit] += 1
            if len(intersection)>0:
                print(intersection)
            isService = sum(service_hits.values()) != 0
            if isService:
                hit_num += 1
                return True

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="facebook/blenderbot-400M-distill",
        #default="facebook/blenderbot-3B",
        type=str,
        help="model to chat with simulator",
    )

    parser.add_argument("--num_chats", default=980, type=int, help="the number of round")

    parser.add_argument("--split", default="train", type=str, help="split")

    parser.add_argument("--seed", default=26, type=int, help="random seed")

    parser.add_argument(
        "--interactive_mode",
        action="store_true",
        help="make the simualtor interact with the user (type 'stop' to stop the program, type 'exit' to leave current round)",
    )

    parser.add_argument(
        "--output",
        default="output.jsonl",
        type=str,
        help="file to save the dialogs",
    )

    parser.add_argument(
        "--disable_output_dialog",
        action="store_true",
        help="whether output the dialogs to the command line",
    )

    args = parser.parse_args()

    return args


def preprocess(example):

    example["personas"] = [f"your persona: {p}" for p in example["personas"]]
    example["context"] = "\n".join(
        example["personas"]
        + (["additional_context"] if example["additional_context"] else [])
        + example["previous_utterance"]
    )

    return example


if __name__ == "__main__":

    args = parse_args()
    random.seed(args.seed)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    with open("keywords.json") as f:
        keywords = json.load(f)
    # lemmatize words in keywords
    for key, val in keywords.items():
        # separate words by its length (one, others)
        one_lemma = []
        multi_lemma = []
        for word in val:
            split = [token.lemma_ for token in nlp(word)]

            if len(split) >= 2:
                multi_lemma.append(" ".join(split))
            else:
                one_lemma.append(split[0])
            keywords[key] = [one_lemma, multi_lemma]
    mname = "facebook/blenderbot-400M-distill"
    simulator = BlenderbotForConditionalGeneration.from_pretrained(mname).to(device)
    simulator_tokenizer = BlenderbotTokenizer.from_pretrained(mname)


    bot = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)
    bot_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dataset = load_dataset("blended_skill_talk", split=args.split)
    dataset = dataset.map(
        preprocess,
        remove_columns=[
            "free_messages",
            "guided_messages",
            "suggestions",
            "personas",
            "additional_context",
            "previous_utterance",
        ],
    )

    if args.interactive_mode:
        for _ in range(args.num_chats):
            dialog = ["hi"]
            while True:
                inputs = simulator_tokenizer(
                    ["</s> <s>".join(dialog[-3:])], return_tensors="pt", truncation=True
                ).to(device)
                reply_ids = simulator.generate(**inputs, do_sample=True, top_p=0.8)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                dialog.append(text)
                print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")

                text = input(f"\033[0;33;49m {'you: ': ^11}")
                dialog.append(text)
                if text in ["stop", "exit"]:
                    break
            if text == "stop":
                break
            print()
    else:
        assert args.num_chats <= len(
            dataset
        ), f"--num_chats needs to be smaller than dataset (<={len(dataset)})"
        dataset = dataset.select(random.sample(range(len(dataset)), args.num_chats))

        output = []
        for index, context in enumerate(
            tqdm(dataset["context"], disable=(not args.disable_output_dialog))
        ):
            print(len(output))
            switch = False
            dialog = []
            if not args.disable_output_dialog:
                print(f" dialog id: {index}")
            for _ in range(5):
                inputs = simulator_tokenizer(
                    [
                        "</s> <s>".join(
                            ([context] + dialog if len(dialog) < 3 else dialog[-3:])
                        )
                    ],
                    return_tensors="pt",
                    truncation=True,
                ).to(device)
                reply_ids = simulator.generate(**inputs)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                dialog.append(text)
                if not args.disable_output_dialog:
                    print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")
                switch = checkhit(dialog,keywords)
                if (switch):
                    break
                # you might need to change this line due to the model you use
                inputs = bot_tokenizer(
                    ["</s> <s>".join(dialog[-3:])], return_tensors="pt", truncation=True
                ).to(device)
                # inputs = bot_tokenizer(
                #     ["[SEP]".join(dialog[-3:])], return_tensors="pt", truncation=True
                # ).to(device)
                # '''
                # input_ids=inputs["input_ids"],
                # attention_mask=inputs["attention_mask"],
                # '''
                # reply_ids = bot.generate(
                #             input_ids=inputs["input_ids"],
                #             attention_mask=inputs["attention_mask"],
                #             )
                reply_ids = bot.generate(
                            **inputs,
                            # num_beams=30, 
                            # no_repeat_ngram_size=2, 
                            )
                
                text = bot_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[
                    0
                ].strip()
                dialog.append(text)
                if not args.disable_output_dialog:
                    print(f"\033[0;33;49m {'bot: ': ^11}{text} \033[0;0m")

            if (switch):
                output.append(dialog)
                pass
            else:
                inputs = simulator_tokenizer(
                    [
                        "</s> <s>".join(
                            ([context] + dialog if len(dialog) < 3 else dialog[-3:])
                        )
                    ],
                    return_tensors="pt",
                    truncation=True,
                ).to(device)
                reply_ids = simulator.generate(**inputs)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                if not args.disable_output_dialog:
                    print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")
                dialog.append(text)
                output.append(dialog)
                if not args.disable_output_dialog:
                    print()


        with open(args.output, "w") as f:
            for idx, dialog in enumerate(output):
                f.write(json.dumps({"id": idx, "dialog": dialog}) + "\n")
