from bpemb import BPEmb

# set up BPE
bpemb = BPEmb(lang='an', vs=100000)

# write all encoded tweets to file
with open('tweets.csv', 'r') as file:
    with open('encodedTweets.csv', 'w') as encoded:
        for tweet in file:
            tweet = bpemb.encode(tweet)
            tweet = ' '.join(tweet)
            encoded.write(tweet)