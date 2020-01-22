import os
import csv
import re
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
# 5) Change ampersands to 'ands' and reduce newlines to spaces#
# 6) Lemmatize all text
# 7) lowercase all text
# 8) Tokenise and pad text

def removeMentions(text):
    return re.sub(r"@\w+:?\s+", "", str(text))

def removeRTs(text):
    return re.sub(r"^RT\s", "", str(text))
    #return re.sub(r"^RT\s@.*:\s", "", text)

file = "./existing_text_test.csv"
df = pd.read_csv(file, header=0)
df["TEXT"] = df["TEXT"].apply(removeRTs).apply(removeMentions)
print(df["TEXT"])
print("\n")
df["TEXT"] = df["TEXT"].replace(r'\n',' ', regex=True)
print(df["TEXT"])
