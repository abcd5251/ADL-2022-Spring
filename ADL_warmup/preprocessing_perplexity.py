import os
import csv

content = []


# sacrebleu ./reference.txt -i generated_predictions.txt -m bleu -b -w 1 --lowercase


with open('./data/out_of_domain/test/source.csv', newline='') as csvfile:

  rows = csv.reader(csvfile)


  with open('./out_predictions.txt', newline='') as f:

        answers = f.readlines()
        
        for index , row in enumerate(rows):
            
            temp = ""
            temp = temp + row[1] + " " + answers[index].strip("\n") + " " + row[2]

            content.append(temp)  

with open('./out_perplexity.txt', 'w') as f:
    for item in content:
        f.write(item + "\n")
