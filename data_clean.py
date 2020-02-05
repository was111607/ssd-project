import re
import numpy as np
import pandas as pd
import pickle
import csv

from collections import Counter
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.corpus import gazetteers
from nltk.corpus import names
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 1) Create dataframe with pandas - use the text only csv?
# 2) Can single out the tweet text column
# 3) Remove RTs, name mentions (@s), punctuation, new lines
# 4) Remove hashtags and space out their components?
# 5) Change ampersands to 'ands'
# 7) lowercase all text
# 8) Tokenise text - remove punctation and lowercase text at the same time
# 6) Lemmatize all text

# 1) Split data into train/test/val sets
# 2) Initialise tokeniser and fit on training set data with fit_on_texts (with oov_token set to True)
# 3) Apply texts_to_sequences on rest of data
# 4) In same method as 3 pre-pad to MAX_SEQ_LENGTH + 1 (First index will be reserved for image sentiments - In future set 0 to "NEG" 1 to "NEU" and 2 to "POS")

# Make file analysing the analytics of the data? Such as vocabulary length, etc...

# Remove sentimental stopwords from stopwords, leaving determiners and conjuncters to be removed from the text:
sntmt_stopwords = {"against", "ain", "aren", "arent", "but", "can", "cant", "cannot", "could", "couldn", "couldnt", "did", "didn", "didnt", "do", "doesn", "does", "doesnt", "doing",
"don", "dont", "few", "had", "hadn", "hadnt", "has", "hasn", "hasnt", "have", "haven", "havent", "having", "hed", "hell", "hes", "id", "is", "ill", "isn", "isnt", "it", "its",
 "ive", "might", "mightn", "mightnt", "mustn", "mustnt", "needn", "neednt", "no", "nor", "not", "shan", "shant", "she", "shed", "shell", "shes", "should", "shouldve", "shouldn",
 "shouldnt", "thatll", "thats", "theres", "theyd", "theyll", "theyre", "theyve", "was", "wasn", "wasnt", "wed", "well", "weve", "were", "weren", "werent", "whats",
 "whens", "wheres", "whos", "whys", "won", "wont", "would", "wouldn", "wouldnt", "youd", "youll", "youre", "youve"}
nltkStopWords = set([re.sub(r"'", "", word) for word in stopwords.words("english")])
externStopWords = set([re.sub(r"'", "", word) for word in get_stop_words("english")])
stopWords = nltkStopWords.union(externStopWords)
stopWords = stopWords - sntmt_stopwords

placeList = set(gazetteers.words())
nameList = set(names.words())
lemmatiser = WordNetLemmatizer()
tokeniser = Tokenizer(oov_token="outofvocab")
counter = Counter()

def removePunct(text):
    rmvApos = re.sub(r"'", "", str(text))
    rmvPunc = re.sub(r"[^\w\s']+", " ", rmvApos) # Whitespace for connecting punctuation
    return rmvPunc.strip()

def removeMentions(text):
    return re.sub(r"@\w+:?\s*", "", str(text))

def removeRTs(text):
    return re.sub(r"^RT\s", "", str(text))
    #return re.sub(r"^RT\s@.*:\s", "", text)

def removeStopWords(text):
    tweet = str(text).split()
    tweet = [word for word in tweet if word not in stopWords]
    return " ".join(tweet)
    # return [word for word in toks if word not in stopWords]

def lemmatise(text):
    tweet = str(text).split()
    tweet = [lemmatiser.lemmatize(word) for word in tweet]
    return " ".join(tweet)
    # return [lemmatiser.lemmatize(word) for word in toks]

def spaceHashes(text):
    tweet = str(text)
    origHashtags = re.findall(r"#\w+", tweet)
    for tag in origHashtags:
        sepTag = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tag)
        sepTag = re.sub(r"([a-z])([0-9])", r"\1 \2", sepTag)
    #    for char in re.findall(r"(?<![A-Z])[A-Z](?![A-Z])", sepTag):
    #        sepTag = sepTag.replace(char, char.lower())
        tweet = tweet.replace(tag, sepTag)
    return tweet

# Remove non-english characters
def removeNEChars(text):
    return re.sub(r"[^[a-zA-Z0-9\s]\w*", "", str(text))

def lowerCase(text):
    tweet = str(text).split()
    normalisedTweet = []
    checkSurname = False
    for word in tweet:
        if (checkSurname) and (re.match(r"[A-Z]\w*", word)):
            normalisedTweet.append(word)
            checkSurname = False
        else:
            checkSurname = False
            if (word in placeList) or (re.match(r"\w*[A-Z]\w*[A-Z]\w*", word)):
                normalisedTweet.append(word)
            elif (word in nameList) or (word[:-1] in nameList and word[len(word) - 1] is "s"):
                normalisedTweet.append(word)
                checkSurname = True
            elif (word[:-1] in placeList) and word[len(word) - 1] is "s":
                normalisedTweet.append(word[:-1])
            else:
                normalisedTweet.append(word.lower())
    return " ".join(normalisedTweet)

def saveData(df, train, test, val):
    with open("existing_text_shuffled", "w") as writeShuff, open("existing_text_train.csv", "w") as writeTrain, open("existing_text_test.csv", "w") as writeTest, open("existing_text_val.csv", "w") as writeVal, open("training_counter.pickle", "wb") as writeCounter:
        df.to_csv(writeShuff, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
        train.to_csv(writeTrain, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
        test.to_csv(writeTest, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
        val.to_csv(writeVal, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
        pickle.dump(counter, writeCounter)
        writeShuff.close()
        writeTrain.close()
        writeTest.close()
        writeVal.close()
        writeCounter.close()

def avgWordCount(df, isBefore):
    avg = df["NEW_TEXT"].apply(lambda x: len(str(x).split())).mean()
    if (isBefore == 1):
        print(f"Average word count before stop-word removal: {avg}") # Find mean word count of text
    else:
        print(f"Average word count after stop-word removal: {avg}") # Find mean word count of text

# Method to find average length of review before stopword removal and after to figure out (Maybe run multiple times??)
def cleanData(df):
    df["NEW_TEXT"] = df["TEXT"]
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removeRTs).apply(removeMentions) # Remove @ mentions and 'RT' text
    df["NEW_TEXT"] = df["NEW_TEXT"].replace(r"\n"," ", regex=True) # Newlines converted into whitespace
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(spaceHashes) # Words within hashtags separated and lowercased
    df["NEW_TEXT"] = df["NEW_TEXT"].replace("&amp;", "and", regex=True) # Replace ampersand with 'and'
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removePunct) # Remove punctuation (including emojis - emulate reviews)
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removeNEChars)
    df["NEW_TEXT"] = df["NEW_TEXT"].replace("\s{2,}", " ", regex=True) # Remove double (or more) spacing
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(lowerCase)
    avgWordCount(df, 1)
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removeStopWords)
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(lemmatise)
    avgWordCount(df, 0)
    return df

def tokeniseTraining(train):
    tweets = list(train["NEW_TEXT"].values)
    for tweet in tweets:
        counter.update(tweet.split())
    tokeniser.fit_on_texts(tweets)
    train["TOKENISED"] = tokeniser.texts_to_sequences(tweets)#train["NEW_TEXT"].apply(tokeniseText)
    train["TOKENISED"] = pad_sequences(train["TOKENISED"], maxlen = 55, padding = "pre", value = 0).tolist() # Converts numpy array to list
    return train

def main():
    file = "./existing_text.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header = 0)
    df = cleanData(df)
    df = df.sample(frac = 1).reset_index(drop = True) # Shuffles data
    train, test, val = np.split(df, [int(.7 * len(df)), int(.9 * len(df))])
    test = test.reset_index(drop = True)
    val = val.reset_index(drop = True)
    train = tokeniseTraining(train)
    saveData(df, train, test, val)

if __name__ == "__main__":
    main()
