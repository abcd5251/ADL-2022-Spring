from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import json
import logging
from pathlib import Path
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, Namespace
from pathlib import Path
import numpy as np 
import csv
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="Path to Glove Embedding.",
        default="./glove.840B.300d.txt"
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Path to path to output predict file.",
        default="./pred_slot.csv"
    )
   

    args = parser.parse_args()
    return args


    

   


def read_json(args):

   
    testing = []
    

    # make test data
    dataset_path = Path(args.test_file)
    dataset = json.loads(dataset_path.read_text())
    logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

    for instance in dataset: # makes test data
             
        temp = [] 
             
        for token in instance["tokens"]:
            temp.append(token)

        testing.append(temp)


    intent2idx = {'date': 1, 'last_name': 2, 'time': 3, 'people': 4, 'first_name': 5, 'O': 0} # for reproduce 
    
    return intent2idx ,testing



def Word2Vector(data_list, word2vec_model):
    """
    look up word vectors
    turn each word into its pretrained word vector
    return a list of word vectors corresponding to each token in train.data
    """

    suffixs = ["’m","'s","’d","'ll","'ve","’s","s'","'ve","'m","'","'re","’ll","’re","!",";","]","驴"]


    x = []
    n = 0
    num = 0
    

    for sentence in data_list:

      vecs = []
      
      for word in  sentence :
       
        try:
          for kk in suffixs :
              word = word.replace(kk, '')

          vec = word2vec_model.get_vector(word)
          vecs.append(vec)
          
        except KeyError:  
          
          if ":" in word:
            vec = word2vec_model.get_vector("pm")
            vecs.append(vec)
          elif "." in word:
            vec = word2vec_model.get_vector("pm")
            vecs.append(vec)
          elif "/" in word:
            vec = word2vec_model.get_vector("february")
            vecs.append(vec)
          elif "pm" in word:
            vec = word2vec_model.get_vector("pm")
            vecs.append(vec)
        
          elif "august" in word:
            vec = word2vec_model.get_vector("august")
            vecs.append(vec)
          else:
            vec = word2vec_model.get_vector("unknown")
            vecs.append(vec)
        
          pass

      num = num+1


      x.append(np.array(vecs))

      

      n += 1
    print("number of sentence :", n )

    return np.array(x)

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
       
        self.fc = nn.Linear(hidden_size , 35)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        # x =  [batch, sequence, channel] if batch_first else [sequence, batch, channel]
        output, _ = self.GRU(x)
        
        y = _.mean(0)

        y =  self.fc(y)

        return y

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
   
    next = True
    with open(file, 'w') as fp:
        
        writer = csv.writer(fp)
        writer.writerow(['id', 'tags'])
        
        for i , num in enumerate(preds):
   
          string = ""
          for j, token in enumerate(num):
            if token == 'O' :
              next = True
              if string=="":

                string = string + token
              else:
                string = string + " "+ token
            else:
              if next == True:
                if string=="":

                  string = string + "B-"+str(token)
                else:
                  string = string + " "+ "B-"+str(token)
                next = False

              elif next == False:

                if string=="":

                  string = string + "B-"+str(token)
                else:
                  if num[j-1] == token:
                    string = string + " "+ "I-"+str(token)
                  else:
                    string = string + " "+ "B-"+str(token)
                
          
          writer.writerow(["test-"+str(i), string])
  
if __name__ == "__main__":

    args = parse_args()
    intent2idx , test_text = read_json(args)
    glove =  KeyedVectors.load_word2vec_format("./gensim_glove.840B.300d.txt", binary=False)
    x_test = Word2Vector(test_text, glove)
    x_test = pad_sequence([torch.from_numpy(np.array(x)) for x in x_test],batch_first = True).float()
    test_dataset = Data_Converter(x_test)
    test_loader = DataLoader(test_dataset, batch_size= 100, shuffle=False, pin_memory=True)


        


    model = My_Model(input_size = 300, hidden_size = 300 ,num_layers = 3 ,batch_first = True).to(device)
    model.load_state_dict(torch.load(args.ckpt_path))
    preds = predict(test_loader, model, device) 
    final = []


    for id , sentence in enumerate(preds) :

        temp_sentence = []
        for i , item in enumerate(sentence):
                    
            if i < len(test_text[id]):

                if item > 0.5 and item <1.5 :

                    temp_sentence.append("date")

                elif item > 1.5 and item <2.5 :

                    temp_sentence.append("last_name")

                elif item > 2.5 and item <3.5 :
                    
                    temp_sentence.append("time")

                elif item > 3.5 and item <4.5 :
                    
                    temp_sentence.append("people")

                elif item > 4.5  :
                
                    temp_sentence.append("first_name")

                else:
                    temp_sentence.append("O")
            else:
                break

        final.append(temp_sentence)

    save_pred(final, args.pred_file)





