from numpy.random import RandomState
import pandas as pd

df = pd.read_csv('tweets.csv')
rng = RandomState()
train = df.sample(frac=0.8, random_state=rng)
test = df.loc[~df.index.isin(train.index)]
train.to_csv(r'train.csv', index=False)
test.to_csv(r'test.csv', index=False)