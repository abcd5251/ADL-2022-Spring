import os
import csv

content = []


# sacrebleu ./reference.txt -i generated_predictions.txt -m bleu -b -w 1 --lowercase


with open('./data/in_domain/test/target.csv', newline='') as csvfile:

  rows = csv.reader(csvfile)

  for row in rows:
    content.append(row[1])

with open('./in_reference.txt', 'w') as f:
    for item in content:
        f.write(item + "\n")

