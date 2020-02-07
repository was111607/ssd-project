import os
import csv
import re
import string
import tweepy as tw # Installed via pip
import pandas as pd
import nltk
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
# from stop_words import get_stop_words
# from nltk.corpus import gazetteers
# from nltk.corpus import stopwords
# from nltk.corpus import names
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import RegexpTokenizer
# from nltk.stem import WordNetLemmatizer


# Load in data as pandas - process images?
# Look into encoding data with one_hot or hashing_trick
# Pad data - find out best pad as it's not 55 - PREPAD, pad as long as longest sequence
# Do text classification and image processing
# When classify together with images, just concatenate vectors together (process images separately)
# Stochastic graident descent?
# split into 70/20/10train test val

# Initially for text
# Objective (optimiser) function, loss function, metrics to define model in Keras
# Investigate activation function
# Embedding layer - LSTM layer - (optional) CNN elements - hidden layer - softmax layer - output layer
# Feature-level fusion: combined image and text vectors within LSTM or before to classify simultaneously
# Decision-level fusion: classify image and text vectors separately, combine within the softmax layer

# model.add(embedding) - size of vocab (retrieve from pickle file) + 1, output dim - have to tinker around, input_length=55 (size of sequences), set mask_zero to true.

def buildModel():
    with open("/media/was/USB DISK/training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 20k
        readFile.close()
    seqLength = 30
    embedDim = 64
    model = Sequential()
    model.add(embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True))

    return None
def main():
    file = "./train_text_input_subset.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header = 0)
    sequences = pd["TOKENISED"]
    model = buildModel()

if __name__ == "__main__":
    main()
