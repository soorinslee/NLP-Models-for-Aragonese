import pandas as pd

with open('vocab.txt', 'r') as file:
    with open('vocab.csv', 'w') as vocab:
        for row in file:
            splits = row.split()
            word = splits[0]

            if ',' in word:
                continue
            else:
                vocab.write(word)
                vocab.write('\n')