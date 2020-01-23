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
# 8) Tokenise text - remove punctation and lowercase text at the same time
# 6) Lemmatize all text

#(?<![A-Z])[A-Z][a-z]+(?![A-Z]+)
def removePunct(text):
    rmvApos = re.sub(r"'", "", str(text))
    return re.sub(r"[^\w\s]+", " ", rmvApos.strip()) # Whitespace for connecting punctuations - removed later anyway

def removeMentions(text):
    return re.sub(r"@\w+:?\s*", "", str(text))

def removeRTs(text):
    return re.sub(r"^RT\s", "", str(text))
    #return re.sub(r"^RT\s@.*:\s", "", text)
def spaceHashes(text):
    tweet = str(text)
    origHashtags = re.findall(r"#\w+", tweet)
    for tag in origHashtags:
        tweet = tweet.replace(tag, re.sub(r"([a-z])([A-Z])", r"\1 \2", tag) # Lowercase individual capital letters SORT THIS OUT IN HASHES

    return tweet

def main():
    file = "./existing_text_test.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header=0)

    df["TEXT"] = df["TEXT"].apply(removeRTs).apply(removeMentions) # Remove @ mentions and 'RT' text
    df["TEXT"] = df["TEXT"].replace(r"\n"," ", regex=True) # Newlines converted into whitespace
    print(df["TEXT"])
    df["TEXT"] = df["TEXT"].apply(spaceHashes) # Words within hashtags separated and lowercased
    print(df["TEXT"])
    df["TEXT"] = df["TEXT"].replace("&amp;", "and", regex=True) # Replace ampersand with 'and'
    df["TEXT"] = df["TEXT"].apply(removePunct) # Remove punctuation (including emojis - emulate reviews)
    print(df["TEXT"])
    df["TEXT"] = df["TEXT"].replace("\s{2,}", " ", regex=True)
    print(df["TEXT"])

if __name__ == "__main__":
    main()
