# R10546017  Homework 2 

## Environment
```shell
# if don't have transformers
pip install transformers
```

## Load Model
```shell
bash download.sh
```

## Run testing for test.json
```shell
# To test the result
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

## Context selection Training
```shell
# Do preprocessing first , make sure train.json and valid.json path are right
python preprocessing_context.py
# After finished
python run_swag.py \--model_name_or_path bert-base-chinese \--train_file train.csv \--validation_file valid.csv \--do_train \--do_eval \--per_device_train_batch_size 3 \--learning_rate 5e-5 \--num_train_epochs 10 \--max_seq_length 512 \--per_gpu_eval_batch_size=3 \--output_dir ./context_valid/ \--overwrite_output
```

## Question answering Training
```shell
# Do preprocessing first , make sure train.json and valid.json path are right
python preprocessing_qa.py
# After finished
python run_qa.py \--model_name_or_path wptoux/albert-chinese-large-qa \--train_file train.csv \--validation_file valid.csv \--do_train \--do_eval \--per_device_train_batch_size 5 \--learning_rate 3e-5 \--num_train_epochs 15 \--max_seq_length 512 \--output_dir ./qa_valid/ \--overwrite_output 
```
