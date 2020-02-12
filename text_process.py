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
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19

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

# Create 3 models separately - image, text embedding and text concat image total model.


# model.add(embedding) - size of vocab (retrieve from pickle file) + 1, output dim - have to tinker around, input_length=55 (size of sequences), set mask_zero to true.
# Add dropout layers into a dense layer with softmax (for neg neu pos classifications) as last layer, takes output of LSTM, maybe add dense before into softmax if needed.
# LOOK INTO RELU, HIDDEN LAYERS = DENSE (ADD RELU HERE)

# Images:
# Either convert optimal caffe model to keras (load weights, create model architecture) or just use VGG19 and predict from keras
# Predict image then, within mode, append to word vector after initial embedding, then reimbed with new dimentions from appending
# Then model proceeds as normal

# Load model without top, flatten output then append to each word vector (total 1920 + 20588 dims)
def loadVGG19(path):
    model = VGG19(weights = "imagenet", include_top = False)
    img = load_img(path, target_size = (224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    #np.nvstack
    featureMatrix = model.predict(img) # (1, 7, 7, 512)

def buildModel():
    with open("/media/was/USB DISK/training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 64
    model = Sequential()
    model.add(Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)) # Output is 30*64 matrix (each word represented in 64 dimensions) = 1920
    model.add(LSTM(embedDim, )) # LSTM, then into softmax, then add ReLU somehere

    return None
def main():
    file = "./train_text_input_subset.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header = 0)
    sequences = pd["TOKENISED"]
    model = buildModel()

if __name__ == "__main__":
    main()
