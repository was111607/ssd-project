import os
import csv
import re
import string
import tweepy as tw # Installed via pip
import pandas as pd
import nltk
from nltk.corpus import stopwords
#from nltk.tokenize import Regexptokenizer
from nltk.stem import WordNetLemmatizer

# 1) Create dataframe with pandas - use the text only csv?
# 2) Can single out the tweet text column
# 3) Remove RTs, name mentions (@s), punctuation, new lines
# 4) Remove hashtags and space out their components?
# 5) Change ampersands to 'ands'
# 7) lowercase all text
# 8) Tokenise text
# 6) Lemmatize all text

def removePunct(text):
    return re.sub(r"[^\w\s]+", "", str(text))

def removeMentions(text):
    return re.sub(r"@\w+:?\s*", "", str(text))

def removeRTs(text):
    return re.sub(r"^RT\s", "", str(text))
    #return re.sub(r"^RT\s@.*:\s", "", text)
def spaceHashes(text):
    tweet = str(text)
    origHashtags = re.findall(r"#\w+", tweet)
    for tag in origHashtags:
        tweet = tweet.replace(tag, re.sub(r"([a-z])([A-Z])", r"\1 \2", tag))
    return tweet
file = "./existing_text_test.csv"
pd.set_option('display.max_colwidth', -1)
df = pd.read_csv(file, header=0)
df["TEXT"] = df["TEXT"].apply(removeRTs).apply(removeMentions)
df["TEXT"] = df["TEXT"].replace(r"\n"," ", regex=True)
print(df["TEXT"])
df["TEXT"] = df["TEXT"].apply(spaceHashes)
print(df["TEXT"])
df["TEXT"] = df["TEXT"].replace("&amp;", "and", regex=True)
df["TEXT"] = df["TEXT"].apply(removePunct)
print(df["TEXT"])
df["TEXT"] = df["TEXT"].replace("\s{2,}", " ", regex=True)
print(df["TEXT"])
#print(spaceHashes("RT @onedirecitiont: Liam deserves #GOON all the love in the world, he's such a cute little bean  #LiamsBirthdayProject"))
