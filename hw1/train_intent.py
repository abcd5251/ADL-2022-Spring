from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import json
import logging
from collections import Counter
from pathlib import Path
from random import random, seed
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder

import os 
import tqdm
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'seed': 520,      
    'batch_size': 100,
    'learning_rate':0.001,
    'n_epochs':1500,
    'data_dir':  "./drive/MyDrive/ADL_hw1/data/intent/",   # Directory to the dataset
    'glove_path': "./drive/MyDrive/ADL_hw1/glove.840B.300d.txt",   # Path to Glove Embedding
    'word2vector_path': "./gensim_glove.840B.300d.txt",     # Directory to save the processed file 
    'save_path': './models/model.ckpt',  
    'early_stop': 200,  
    'valid_ratio': 0.2     
}


def construct_word2vec(): 

    (count, dimensions) = glove2word2vec(config['glove_path'], config['word2vector_path'])
    print(count, '\n', dimensions)
    model = KeyedVectors.load_word2vec_format(config['word2vector_path'], binary=False)

    return model

def read_json():

    intents = [] 
    texts = []
    testing = []
    labels = set()
    maxi = 0

    for split in ["train", "eval"]:

        dataset_path = Path( config['data_dir'] + f"{split}.json")
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        for instance in dataset: # makes train data
             temp = [] 
             intents.append(instance["intent"])
             labels.add(instance["intent"]) # makes set for intent2idx 
             
             number_of_word = 0

             for token in instance["text"].split():
                 temp.append(token)
                 number_of_word = number_of_word + 1 

             if number_of_word> maxi :
                  maxi = number_of_word

             texts.append(temp)

    # make test data
    dataset_path = Path( config['data_dir'] + "test.json")
    dataset = json.loads(dataset_path.read_text())
    logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

    for instance in dataset: # makes test data
             
        temp = [] 
             
        for token in instance["text"].split():
            temp.append(token)

        testing.append(temp)



    intent2idx = {tag: i for i, tag in enumerate(labels)}
    
    for index , tt in enumerate(intents):
        intents[index] = intent2idx[tt]
    
    
   

    return intents , texts , intent2idx , maxi ,testing



train_intents , train_texts , intent2idx , maxi_text , test_text = read_json()
glove = construct_word2vec()

train_label = []

for i in train_intents:
    count = 0
    temp = []
    while count < 150:

      if count ==i:
        temp.append(1)
      else:
        temp.append(0)
      count = count + 1

    train_label.append(temp)

print(train_intents)
print(train_label[7])

import numpy as np 
from sklearn.model_selection import train_test_split
#from nltk.stem import WordNetLemmatizer 
#import nltk
#nltk.download('wordnet')
#from nltk.stem.porter import PorterStemmer
#stemmer=PorterStemmer()
#lemmatizer = WordNetLemmatizer()

def Word2Vector(data_list, word2vec_model, maxi_len):
    """
    look up word vectors
    turn each word into its pretrained word vector
    return a list of word vectors corresponding to each token in train.data
    """

    suffixs = ["’m","'s",",","'d",":","'ll","!",";","'ve","’s","s'","'ve","'m","'","$","ab123","(",")",'"',"'re","’ll","’re"]

    v = word2vec_model.get_vector('king')
    dim  = len(v)


    x = []
    n = 0
   

    for sentence in data_list:

      vecs = []
      
      for word in  sentence :

        #word = lemmatizer.lemmatize(word)   not even better
       
        try:
          for kk in suffixs :
              word = word.replace(kk, '')

          vec = word2vec_model.get_vector(word)
          vecs.append(vec)
          
        except KeyError:          
                 
          pass


      x.append(np.array(vecs))

      

      n += 1
    print("number of sentence :", n )

    return np.array(x)


x_train, x_val, y_train, y_val = train_test_split( train_texts, train_label , test_size=config["valid_ratio"], random_state=config["seed"])


x_train = Word2Vector(x_train, glove , maxi_text)
x_val = Word2Vector(x_val, glove , maxi_text)
x_test = Word2Vector(test_text, glove , maxi_text)

x_train = pad_sequence([torch.from_numpy(np.array(x)) for x in x_train],batch_first = True).float()
x_val = pad_sequence([torch.from_numpy(np.array(x)) for x in x_val],batch_first = True).float()
x_test = pad_sequence([torch.from_numpy(np.array(x)) for x in x_test],batch_first = True).float()

print(len(x_train))
print(len(x_val))
print(len(x_test))
def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

print("yes")

def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.CrossEntropyLoss(reduction='mean') 

  
    optimizer = torch.optim.RAdam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
   
  

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)        
            
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        
        mean_train_loss = sum(loss_record)/len(loss_record)
        

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)                
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)

        
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return best_loss
    return best_loss

class Data_Converter(Dataset):

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


train_dataset, valid_dataset, test_dataset = Data_Converter(x_train, y_train), \
                                            Data_Converter(x_val, y_val), \
                                            Data_Converter(x_test)
                                            
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class My_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, drop=0.3):
        super(My_Model, self).__init__()
        self.GRU = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=batch_first,
            dropout=drop,
            bidirectional=True,
        )
        #self.linear = nn.Linear(hidden_size , 150)
        self.fc = nn.Linear(hidden_size , 150)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        # x =  [batch, sequence, channel] if batch_first else [sequence, batch, channel]
        output, _ = self.GRU(x)
        
        y = _.mean(0)
        #y = self.linear(y)
        y =  self.fc(y)

        # y = self.softmax(y) 
        return y

same_seed(config['seed'])

model = My_Model(input_size=len(x_train[0][0]), hidden_size = 300 ,num_layers = 2 ,batch_first = True).to(device) # put your model and data on the same computation device.
best_loss = trainer(train_loader, valid_loader, model, config, device)
print("best_loss :" + str(best_loss))


import csv

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'intent'])
        for i, p in enumerate(preds):
            writer.writerow(["test-"+str(i), p])




model = My_Model(input_size=len(x_train[0][0]), hidden_size = 300 ,num_layers = 2 ,batch_first = True).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device) 




final = []
list_of_value = list(intent2idx.values())
list_of_key = list(intent2idx.keys())

for item in preds :
    position = list_of_value.index(np.argmax(item))
    final.append(list_of_key[position])

save_pred(final, './pred.csv')
