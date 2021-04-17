from bpemb import BPEmb

# set up BPE
bpemb = BPEmb(lang='an', vs=100000)

# write all encoded train tweets to file
with open('train.csv', 'r') as file:
    with open('encodedTrain.csv', 'w') as encoded:
        for tweet in file:
            tweet = bpemb.encode(tweet)
            tweet = ' '.join(tweet)
            encoded.write(tweet)