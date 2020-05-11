"""
--------------------------
Written by William Sewell
--------------------------
Performs the text preprocessing functionalities.

This step was executed using the local machine.

---------------
Files Required
---------------
existing_text.csv - Stores ID, sentiment polarities and text content for all
                    existing tweets from data collation.

b-t4sa_train.txt - Stores the image paths and classified text sentiments for the BT4SA training split.

b-t4sa_val.txt - Stores the image paths and classified text sentiments for the BT4SA validation split.

b-t4sa_test.txt - Stores the image paths and classified text sentiments for the BT4SA testing split.

---------------
Files Produced
---------------
existing_text_train.csv - Stores the training split data for existing_text.csv
                          adding columns storing the filtered and tokenised text.

existing_text_val.csv - Stores the validation split data for existing_text.csv
                        adding columns storing the filtered and tokenised text.

existing_text_test.csv - Stores the testing split data for existing_text.csv
                         adding columns storing the filtered and tokenised text.

vocabulary.pickle - Stores the recorded vocabulary on the entire data set.
"""

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
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Remove sentimental stopwords from stopwords, leaving determiners and conjuncters to be removed from the text:
# Define all sentimental stopwords as a set.
sntmt_stopwords = {"against", "ain", "aren", "arent", "but", "can", "cant", "cannot", "could", "couldn", "couldnt", "did", "didn", "didnt", "do", "doesn", "does", "doesnt", "doing",
"don", "dont", "few", "had", "hadn", "hadnt", "has", "hasn", "hasnt", "have", "haven", "havent", "having", "hed", "hell", "hes", "id", "is", "ill", "isn", "isnt", "it", "its",
 "ive", "might", "mightn", "mightnt", "mustn", "mustnt", "needn", "neednt", "no", "nor", "not", "shan", "shant", "she", "shed", "shell", "shes", "should", "shouldve", "shouldn",
 "shouldnt", "thatll", "thats", "theres", "theyd", "theyll", "theyre", "theyve", "was", "wasn", "wasnt", "wed", "well", "weve", "were", "weren", "werent", "whats",
 "whens", "wheres", "whos", "whys", "won", "wont", "would", "wouldn", "wouldnt", "youd", "youll", "youre", "youve"}
# Load NLTK and Python-stop-words as sets to perform set operations for efficiency.
nltkStopWords = set([re.sub(r"'", "", word) for word in stopwords.words("english")]) # Remove apostrophes to match sntmt_stopwords format.
externStopWords = set([re.sub(r"'", "", word) for word in get_stop_words("english")])
stopWords = nltkStopWords.union(externStopWords) # Set union on all required stopwords.
stopWords = stopWords - sntmt_stopwords # Set difference to remove sentimental stopwords from all stopwords.

# Establishes place and name list for efficiency in their removal from the text content.
placeList = set(gazetteers.words())
nameList = set(names.words())

# Initialises lemmatiser, tokeniser and vocabulary-recording objects.
lemmatiser = WordNetLemmatizer()
tokeniser = Tokenizer(oov_token="outofvocab") # Tokeniser will replace unseen words with oov_token.
counter = Counter() # Initialise vocabulary counter.

# Remove punctuation from recieved tweet text.
def removePunct(text):
    # Allows separated components from apostrophes to form a single word e.g. youre.
    rmvApos = re.sub(r"'", "", str(text))
    # Match on non-word or emoji unicode characters.
    rmvPunc = re.sub(r"[^\w\s\U00010000-\U0010ffff\U00002764]+", " ", rmvApos, flags=re.UNICODE)
    return rmvPunc.strip() # Removes leading and trailing characters.

# Remove user mentions from the tweet.
# User mentions are always formatted as characters following '@' and preceding ':'.
def removeMentions(text):
    return re.sub(r"@\w+:?\s*", "", str(text)) # Replaces matched characters with empty string to 'remove'.

# Removes the text 'RT' and its following space from the beginning of a tweet that
# signifies a retweet and is followed by a user mention
def removeRTs(text):
    return re.sub(r"^RT\s", "", str(text))

# Removes non-sentimental stopwords from received tweet text.
# Converts the text into a vector to apply stopword removal for efficiency.
def removeStopWords(text):
    tweet = str(text).split() # Converts text into a list storing individual words, originally separated by whitespace.
    tweet = [word for word in tweet if word not in stopWords] # Constructs a list of words found not to be a stopword via iteration through the vector.
    return " ".join(tweet) # Reconstructs vector as string separating words by whitespace.

# Performs word lemmatisation on tweet text.
# Converts the text into a vector to apply lemmatisation for efficiency.
def lemmatise(text):
    tweet = str(text).split() # Converts text into a list storing individual words, originally separated by whitespace.
    tweet = [lemmatiser.lemmatize(word) for word in tweet] # Applies lemmatisation iteratively to each word in list.
    return " ".join(tweet) # Reconstructs vector as string separating words by whitespace.

# Spaces emojis apart.
# Each emoji character that matches the unicdoe is placed in its own 'group'
# so it can be preserved and whitespace simply added around it and returned.
def spaceEmojis(text):
    return re.sub(r"([\U00010000-\U0010ffff\U00002764])", r" \1 ", str(text), flags=re.UNICODE)

# Spaces keywords inside hashtags.
# Can only space on capitalised letters as this is an acceptable way of easy identification.
def spaceHashes(text):
    tweet = str(text)
    origHashtags = re.findall(r"#\w+", tweet) # Produces list containing all found hashtags.
    # Iterate through each hashtag separating keywords.
    for tag in origHashtags:
        sepTag = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tag) # Finds first found capitalised letter and ending character for the word before it
                                                             # groups them into individual groups such that whitespace is added between.
        sepTag = re.sub(r"([a-z])([0-9])", r"\1 \2", sepTag) # Then, the above process is applied but separating numbers from word character
                                                             # endings.
        tweet = tweet.replace(tag, sepTag) # Replace original hashtag string in text with the character separation results.
    return tweet

# Remove non-english characters.
# Regular expression identifies all character collections containing non-english
# or non-emoji characters.
def removeNEChars(text):
    return re.sub(r"\w*[^a-zA-Z0-9\s\U00010000-\U0010ffff\U00002764]\w*", "", str(text), flags=re.UNICODE)

# Lowercases words with the exception of:
# - Common names and accompanying surnames (if applicable)
# - Common place names
# - Places or names with the ownership 's' e.g. 'Russias'
def lowerCase(text):
    tweet = str(text).split() # Converts text to a vector of separate words for efficiency.
    normalisedTweet = [] # Stores a list of words that have been processed.
    checkSurname = False # If true, triggers word checks for surname.
    for word in tweet:
        # If the previous word was matched first name and the current word is capitalised, it is assumed to be a surname.
        if (checkSurname) and (re.match(r"[A-Z]\w*", word)):
            normalisedTweet.append(word)
            checkSurname = False # Resets checker.
        else:
            checkSurname = False
            # If the word is matched against a place from the corpus or contains at least 2 capitalised letters,
            # the unmodified word is retained.
            if (word in placeList) or (re.match(r"\w*[A-Z]\w*[A-Z]\w*", word)):
                normalisedTweet.append(word)
            # If a word entirely matches against a name from the corpus,
            # or with the last letter removed to check if it is an 's' as this indicates ownership,
            # the unmodified word is retained and the next word will be checked to see if it is a surname.
            elif (word in nameList) or (word[:-1] in nameList and word[len(word) - 1] == "s"):
                normalisedTweet.append(word)
                checkSurname = True
            # If a word with the last letter removed matches against a place name
            # and the last letter is an 's', indicating ownershp,
            # the word with removed ownership is retained.
            elif (word[:-1] in placeList) and (word[len(word) - 1] == "s"):
                normalisedTweet.append(word[:-1])
            # Otherwise the lowercased word is used.
            else:
                normalisedTweet.append(word.lower())
    return " ".join(normalisedTweet) # Returns list as a string with words separated by whitespace.

# CSVs are initialised and the text rows corresponding to each split are saved into them.
# index = False means that the row indexes for the data is not added to the csv.
# The vocabulary is also saved.
def saveData(train, test, val):
    with open("./b-t4sa/existing_text_train.csv", "w") as writeTrain, open("./b-t4sa/existing_text_test.csv", "w") as writeTest, open("./b-t4sa/existing_text_val.csv", "w") as writeVal, open("vocabulary.pickle", "wb") as writeCounter:
        train.to_csv(writeTrain, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        test.to_csv(writeTest, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        val.to_csv(writeVal, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        pickle.dump(counter, writeCounter)
        writeTrain.close()
        writeTest.close()
        writeVal.close()
        writeCounter.close()

# Prints mean word count across all texts at current point.
# isBefore determines if the function is called before or after stop-word removal,
# to assess the length to set tokenised vectors to, and is displayed accordingly.
def avgWordCount(df, isBefore):
    # Calculates mean of text data, which is converted to list format to calculate their length.
    avg = df["NEW_TEXT"].apply(lambda x: len(str(x).split())).mean()
    if (isBefore == 1):
        print(f"Average word count before stop-word removal: {avg}")
    else:
        print(f"Average word count after stop-word removal: {avg}")

# Main preprocess function that applies filtering in a series of steps to the
# "NEW_TEXT" column of the passed in DataFrame.
def preProcess(df):
    df["NEW_TEXT"] = df["TEXT"]
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removeRTs).apply(removeMentions) # Remove @ mentions and 'RT' text together.
    df["NEW_TEXT"] = df["NEW_TEXT"].replace(r"\n"," ", regex=True) # Newlines are converted into whitespace.
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(spaceHashes)
    df["NEW_TEXT"] = df["NEW_TEXT"].replace("&amp;", "and", regex=True) # Replace ampersand with 'and'.
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removePunct) # Remove punctuation.
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(spaceEmojis)
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removeNEChars)
    df["NEW_TEXT"] = df["NEW_TEXT"].replace("\s{2,}", " ", regex=True) # Remove double (or more) spacing.
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(lowerCase)
    avgWordCount(df, 1)
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(removeStopWords)
    df["NEW_TEXT"] = df["NEW_TEXT"].apply(lemmatise)
    avgWordCount(df, 0)
    return df

# Tokeniser method that iteratively converts filtered text data, passed in as a numpy array and converted to a list,
# of its words from the "NEW_TEXT" DataFrame that stores the column of filtered text data.
# The vocabubulary updates on each tweet's words and the tokeniser fits on the training
# data only to unbiasedly represent the rest of the data.
# All tweet word lists are converted to a token vector using the tokeniser and then prepadded
# with 0 values to unify the lengths.
# The results are returned as a column storing lists.
def tokenise(df, isTrain):
    tweets = list(df["NEW_TEXT"].values)
    for tweet in tweets:
        counter.update(tweet.split()) # update vocabulary
    if (isTrain is True):
        tokeniser.fit_on_texts(tweets)
    df["TOKENISED"] = tokeniser.texts_to_sequences(tweets) # Stores tokenised text as a vector of tokens into a new column.
    df["TOKENISED"] = pad_sequences(df["TOKENISED"], maxlen = 30, padding = "pre", value = 0).tolist() # Converts numpy array to list for CSV storage.
    return df

# Additional unused function that can randomly divide unsplitted data into training, validation and test splits.
def randomSplit(df):
    train, test, val = np.split(df, [int(.7 * len(df)), int(.9 * len(df))]) # Takes indexes to divide the splits between
    test = test.reset_index(drop = True)
    val = val.reset_index(drop = True)
    return train, test, val

# Returns a list of all the tweet IDS in the passed b-t4sa data splits.
def getIDList(fname):
    idList = []
    with open(fname + ".txt", "r") as readFile:
        for line in readFile:
            # Matches on tweet ID portion inside the image path in the file line.
            idList.append(re.search(r"(?<=/)[0-9]+(?=-)", line).group(0))
        readFile.close()
    return idList

# Reflects BT4SA training, test and validation splits
# across the entire DataFrame by producing separate DataFrames
# containing the rows belong to each split by their ID, produced calling
# getIDList.
def bt4saExistSplits(df):
    idList = getIDList("./b-t4sa/b-t4sa_train")
    train = df[df["TWID"].isin(idList)] # only populates DataFrame with rows corresponding to ID's matching those in ID list.
    idList = getIDList("./b-t4sa/b-t4sa_test")
    test = df[df["TWID"].isin(idList)]
    idList = getIDList("./b-t4sa/b-t4sa_val")
    val = df[df["TWID"].isin(idList)]
    return train, test, val

def main():
    file = "./existing_text.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header = 0) # Removes row containing column headings.
    df = preProcess(df) # Filters all rows.
    df = df.sample(frac = 1).reset_index(drop = True) # Shuffles data.
    train, test, val = bt4saExistSplits(df)
    # Tokenise filtered text data in rows for each DataFrame split.
    train = tokenise(train, True)
    test = tokenise(test, False)
    val = tokenise(val, False)
    saveData(train, test, val)

if __name__ == "__main__":
    main()
