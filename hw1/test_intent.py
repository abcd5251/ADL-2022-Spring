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
import csv
import numpy as np
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
        default="pred_intent.csv"
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
             
        for token in instance["text"].split():
            temp.append(token)

        testing.append(temp)

    
   

    return testing

def Word2Vector(data_list, word2vec_model):
    """
    look up word vectors
    turn each word into its pretrained word vector
    return a list of word vectors corresponding to each token in train.data
    """

    suffixs = ["’m","'s",",","'d",":","'ll","!",";","'ve","’s","s'","'ve","'m","'","$","ab123","(",")",'"',"'re","’ll","’re"]

    x = []
    n = 0 

    for sentence in data_list:

      vecs = []

      for word in  sentence :

       
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
        self.fc = nn.Linear(hidden_size , 150)
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
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'intent'])
        for i, p in enumerate(preds):
            writer.writerow(["test-"+str(i), p])

if __name__ == "__main__":

    args = parse_args()
    test_text = read_json(args)
    glove = KeyedVectors.load_word2vec_format("./gensim_glove.840B.300d.txt", binary=False)
    x_test = Word2Vector(test_text, glove)
    x_test = pad_sequence([torch.from_numpy(np.array(x)) for x in x_test],batch_first = True).float()

    test_dataset = Data_Converter(x_test)
    test_loader = DataLoader(test_dataset, batch_size= 100, shuffle=False, pin_memory=True)

    model = My_Model(input_size= 300, hidden_size = 300 ,num_layers = 2 ,batch_first = True).to(device)
    model.load_state_dict(torch.load(args.ckpt_path))
    preds = predict(test_loader, model, device) 

    # for reproduce 

    intent2idx =  {'rollover_401k': 0, 'balance': 1, 'spending_history': 2, 'plug_type': 3, 'who_made_you': 4, 'cancel': 5, 'timezone': 6, 'w2': 7, 'calendar': 8, 'change_accent': 9, 'pay_bill': 10, 'insurance_change': 11, 'interest_rate': 12, 'carry_on': 13, 'meal_suggestion': 14, 'order': 15, 'schedule_maintenance': 16, 'restaurant_suggestion': 17, 'calendar_update': 18, 'book_flight': 19, 'transactions': 20, 'schedule_meeting': 21, 'pin_change': 22, 'travel_notification': 23, 'shopping_list': 24, 'meeting_schedule': 25, 'mpg': 26, 'translate': 27, 'redeem_rewards': 28, 'change_speed': 29, 'find_phone': 30, 'income': 31, 'whisper_mode': 32, 'goodbye': 33, 'recipe': 34, 'insurance': 35, 'text': 36, 'restaurant_reviews': 37, 'yes': 38, 'lost_luggage': 39, 'ingredients_list': 40, 'tire_pressure': 41, 'last_maintenance': 42, 'reminder_update': 43, 'bill_balance': 44, 'how_busy': 45, 'report_fraud': 46, 'vaccines': 47, 'direct_deposit': 48, 'gas': 49, 'ingredient_substitution': 50, 'are_you_a_bot': 51, 'tire_change': 52, 'what_song': 53, 'report_lost_card': 54, 'pto_used': 55, 'flip_coin': 56, 'calculator': 57, 'date': 58, 'pto_request': 59, 'sync_device': 60, 'travel_alert': 61, 'transfer': 62, 'todo_list_update': 63, 'accept_reservations': 64, 'travel_suggestion': 65, 'expiration_date': 66, 'min_payment': 67, 'credit_limit_change': 68, 'application_status': 69, 'book_hotel': 70, 'confirm_reservation': 71, 'restaurant_reservation': 72, 'pto_request_status': 73, 'definition': 74, 'change_language': 75, 'next_song': 76, 'reminder': 77, 'credit_limit': 78, 'oil_change_when': 79, 'international_fees': 80, 'flight_status': 81, 'measurement_conversion': 82, 'todo_list': 83, 'improve_credit_score': 84, 'new_card': 85, 'no': 86, 'weather': 87, 'how_old_are_you': 88, 'meaning_of_life': 89, 'bill_due': 90, 'taxes': 91, 'directions': 92, 'freeze_account': 93, 'smart_home': 94, 'maybe': 95, 'exchange_rate': 96, 'payday': 97, 'traffic': 98, 'roll_dice': 99, 'international_visa': 100, 'credit_score': 101, 'food_last': 102, 'what_is_your_name': 103, 'jump_start': 104, 'where_are_you_from': 105, 'fun_fact': 106, 'oil_change_how': 107, 'change_volume': 108, 'replacement_card_duration': 109, 'account_blocked': 110, 'spelling': 111, 'tell_joke': 112, 'calories': 113, 'order_status': 114, 'cook_time': 115, 'next_holiday': 116, 'what_can_i_ask_you': 117, 'play_music': 118, 'time': 119, 'gas_type': 120, 'card_declined': 121, 'apr': 122, 'share_location': 123, 'nutrition_info': 124, 'alarm': 125, 'reset_settings': 126, 'rewards_balance': 127, 'do_you_have_pets': 128, 'thank_you': 129, 'shopping_list_update': 130, 'timer': 131, 'order_checks': 132, 'greeting': 133, 'what_are_your_hobbies': 134, 'user_name': 135, 'change_ai_name': 136, 'current_location': 137, 'change_user_name': 138, 'routing': 139, 'repeat': 140, 'make_call': 141, 'damaged_card': 142, 'pto_balance': 143, 'update_playlist': 144, 'cancel_reservation': 145, 'uber': 146, 'who_do_you_work_for': 147, 'distance': 148, 'car_rental': 149}
    

    final = []
    list_of_value = list(intent2idx.values())
    list_of_key = list(intent2idx.keys())

    for item in preds :
        position = list_of_value.index(np.argmax(item))
        final.append(list_of_key[position])

    save_pred(final, args.pred_file)

