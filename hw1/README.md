# R10546017  Homework 1 

## Environment
```shell
# if don't have genism
pip install gensim
# If don't have glove.840B.300d.zip or no word2vector's glove
bash preprocess.sh
```

## Load Model
```shell
bash download.sh
```

## Intent Classification Testing
```shell
# To test Intent detection
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot Tagging Testing
```shell
# To test Slot detection
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
```

## Intent Classification Training
```shell
python train_intent.py
# otherwise 
can use train_intent.ipynb to check
```

## Slot Tagging Training
```shell
python train_slot.py
# otherwise 
can use train_slot.ipynb to check
```
