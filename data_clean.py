import os
import csv
import re
import string
import tweepy as tw # Installed via pip
import pandas as pd
import nltk
from nltk.corpus import gazetteers
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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
placesList = set(gazetteers.words())
stopwords = set([re.sub(r"'", "", word) for word in stopwords.words("english")])
lemmatiser = WordNetLemmatizer()
#set(re.sub(r"'", "", stopwords.words("english")))
#print(stopwords)

def removePunct(text):
    rmvApos = re.sub(r"'", "", str(text))
    rmvPunc = re.sub(r"[^\w\s']+", " ", rmvApos) # Whitespace for connecting punctuations - removed later anyway
    return rmvPunc.strip()

def removeMentions(text):
    return re.sub(r"@\w+:?\s*", "", str(text))

def removeRTs(text):
    return re.sub(r"^RT\s", "", str(text))
    #return re.sub(r"^RT\s@.*:\s", "", text)

def removeStopWords(toks):
    return [word for word in toks if word not in stopwords]
def lemmatise(toks):
    return [lemmatiser.lemmatize(word) for word in toks]

def spaceHashes(text):
    tweet = str(text)
    origHashtags = re.findall(r"#\w+", tweet)
    for tag in origHashtags:
        sepTag = re.sub(r"([a-z])([A-Z])", r"\1 \2", tag)
        for char in re.findall(r"(?<![A-Z])[A-Z](?![A-Z])", sepTag):
            sepTag = sepTag.replace(char, char.lower())
        tweet = tweet.replace(tag, sepTag) # Lowercase individual capital letters SORT THIS OUT IN HASHES
    return tweet

def lowerCase(text):
    tweet = str(text).split()
    normalisedTweet = []
    for word in tweet:
        if not (word in placesList) and not re.match(r"\w*[A-Z]\w*[A-Z]\w*", word):
            normalisedTweet.append(word.lower())
        else:
            normalisedTweet.append(word)
    return " ".join(normalisedTweet)

def saveData(df):
    with open ("existing_text_tokenised.csv", "w") as writeFile:
        df.to_csv(writeFile, index=False)
        writeFile.close()

def main():
    file = "./existing_text.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header=0)
    df["NEW_TEXT"] = df["TEXT"]
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removeRTs).apply(removeMentions) # Remove @ mentions and 'RT' text
    df["NEW_TEXT"] = df["NEW_TEXT"].replace(r"\n"," ", regex=True) # Newlines converted into whitespace
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(spaceHashes) # Words within hashtags separated and lowercased
    df["NEW_TEXT"] = df["NEW_TEXT"].replace("&amp;", "and", regex=True) # Replace ampersand with 'and'
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removePunct) # Remove punctuation (including emojis - emulate reviews)
    df["NEW_TEXT"] = df["NEW_TEXT"].replace("\s{2,}", " ", regex=True) # Remove double (or more) spacing
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(lowerCase)
    df["TOK_TEXT"] = df["NEW_TEXT"].apply(word_tokenize)
    df["TOK_TEXT"] = df["TOK_TEXT"].apply(removeStopWords)
    df["TOK_TEXT"] = df["TOK_TEXT"].apply(lemmatise)
    saveData(df)

if __name__ == "__main__":
    main()
