# R10546017  Homework 3 

## Environment
```shell
# if don't have ckiptagger
pip install ckiptagger
# if don't have transformers
pip install transformers
# if don't have rouge
pip install rouge
```

## Load Model
```shell
# download rouge for tw-rouge and word segmentation ( WS )
# download model weight 
bash ./download.sh
# make sure ckiptagger's data is in the right place
# default ./data
```

## Run testing for data
```shell
# To test the result
# Make sure model weight is in ./model.ckpt
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

## Title Generation Training
```shell
# need to modify config in run_train.py for the right path
python run_train.py
```