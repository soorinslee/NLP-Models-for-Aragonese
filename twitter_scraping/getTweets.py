import tweepy
import re

# https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
# get rid of emojis in text
def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"
                                        u"\U0001F300-\U0001F5FF"
                                        u"\U0001F680-\U0001F6FF"
                                        u"\U0001F1E0-\U0001F1FF"
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001f926-\U0001f937"
                                        u'\U00010000-\U0010ffff'
                                        u"\u200d"
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\u3030"
                                        u"\ufe0f"
                                        "]+", flags=re.UNICODE)

    return regrex_pattern.sub(r'', text)


# Twitter API
consumer_key = 'CONSUMER_KEY'
consumer_secret = 'SECRET_CONSUMER_KEY'
access_token = 'ACCESS_TOKEN'
access_token_secret = 'SECRET_ACCESS_TOKEN'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# get tweets from top % Aragonese Twitter users
top1 = tweepy.Cursor(api.user_timeline, id="ChodorowskyAra")
top2 = tweepy.Cursor(api.user_timeline, id="ArredolToday")
top3 = tweepy.Cursor(api.user_timeline, id="PodemosZgz_Ara")
top4 = tweepy.Cursor(api.user_timeline, id="zgzencomun")
top5 = tweepy.Cursor(api.user_timeline, id="ixutomullamuto")
top6 = tweepy.Cursor(api.user_timeline, id="arredol")
top7 = tweepy.Cursor(api.user_timeline, id="orachezaragoza")
top8 = tweepy.Cursor(api.user_timeline, id="CantaenAragones")
top9 = tweepy.Cursor(api.user_timeline, id="chabimp")
top10 = tweepy.Cursor(api.user_timeline, id="troballa")

# list of unwanted characters
unwanted = ['#', '@', '.', '"', '/', ',', '?', '!', ';', ':', '-', '(', ')' '_', '?',
            '¿', '¡', '%', '&', '*', '+', '=', '$', '^', '~', '[', ']', '{', '}',
            '|', '\\', '>', '<', 'º', '€']

# write tweets to file
with open('tweets.csv', 'w') as file:
    for status in top1.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')

    for status in top2.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')

    for status in top3.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')

    for status in top4.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')

    for status in top5.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')

    for status in top6.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')

    for status in top7.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')

    for status in top8.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')

    for status in top9.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')

    for status in top10.items():
        tweets = re.split(r'(?<=[.!?]) +', status.text)

        for tweet in tweets:
            tweet = deEmojify(tweet)
            tweet = tweet.lower()
            # get rid of URLs
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'www\S+', '', tweet)
            # get rid of degrees sign
            tweet = tweet.replace(u'\N{DEGREE SIGN}', '')
            # get rid of numbers
            tweet = ''.join([i for i in tweet if not i.isdigit()])
            # get rid of unwanted characters
            tweet = ''.join([i for i in tweet if not i in unwanted])
            # strip newline
            tweet = tweet.strip('\n')

            # skip to next tweet if there are no letters
            if re.search('[a-zA-Z]', tweet) is None:
                continue

            if len(tweet.split()) == 0:
                continue

            file.write(tweet)
            file.write('\n')