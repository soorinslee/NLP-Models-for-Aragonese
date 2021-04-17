import random
from bpemb import BPEmb

# set up BPE
bpemb = BPEmb(lang='an', vs=100000)

# write all encoded masked tweets to file
with open('encodedTweets.csv', 'r') as file:
    with open('encodedMasked.csv', 'w') as masked:
        for row in file:
            i = random.randint(0, len(row.split()) - 1)
            splits = row.split();
            splits[i] = '[MASK]'
            sentence = ' '.join(splits)
            masked.write(sentence)

            if '\n' not in sentence:
                masked.write('\n')