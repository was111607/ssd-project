import os
import csv
import re
import string
import tweepy as tw # Installed via pip
import pandas as pd
import nltk
import pickle
import numpy as np
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Dropout
from keras.layers.merge import add
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from ast import literal_eval
# Load in data as pandas - process images?
# Look into encoding data with one_hot or hashing_trick
# Pad data - find out best pad as it's not 55 - PREPAD, pad as long as longest sequence
# Do text classification and image processing
# When classify together with images, just concatenate vectors together (process images separately)
# Stochastic graident descent?
# split into 70/20/10train test val
# HyperParameter optimisation

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
counter = 0
def getImageRep(path):
    global counter
    print(counter)
    counter += 1
    img = load_img(str(path), target_size = (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def getImgReps(df): # pathList old arg
    images = []
    vgg19 = VGG19(weights = "imagenet")
    model = Sequential()
    for layer in vgg19.layers[:-1]: # Output of FC2 layer
        model.add(layer)
    model.add(Dense(512, activation = "relu"))
    df["REPRESENTATION"] = df.apply(getImageRep)

#    firstImg = None
    # pathListLen = len(pathList)
    # input(len(pathList))
    #featureMatrix = np.empty(pathListLen, dtype = object)
    #for i in range(pathListLen):
    # for path in pathList:
    #     print(counter)
    # #    img = load_img(pathList[i], target_size = (224, 224))
    #     #featureMatrix[i] = img
    #     # img = load_img(path, target_size = (224, 224))
    #     # img = img_to_array(img)
    #     # img = np.expand_dims(img, axis=0)
    #     if (len(images) == 0):
    #         if (firstImg is None):
    #             firstImg = img
    #         else:
    #             images = np.vstack([firstImg, img])
    #     else:
    #         images = np.vstack([images, img])
    #     counter += 1
    # featureMatrix = np.concatenate(featureMatrix)
    # featureMatrix = model.predict(img) # (x, 512)
    # return featureMatrix
    return df

# numarray = np.array([np.arange(1, 513), np.arange(1, 513)])
# vgg19 = VGG19(weights='imagenet')
# # model = Sequential()
# # for layer in vgg19.layers[:-1]:
# #     model.add(layer)
# reduceImgFtrs = Dense(512, activation = "relu")(vgg19.layers[-2].output)
# textFtrs = Input(shape=(512,))
# added = add([reduceImgFtrs, textFtrs])
# model = Model(inputs=[vgg19.input, textFtrs], output=added)

def mainModel():
    with open("/media/was/USB DISK/training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    #textFtrs = Dense(embedDim, use_bias = False)(textFtrs)
    #print(textFtrs.output)
    imageFtrs = Input(shape=(embedDim,))
    added = add([textFtrs, imageFtrs])
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.2, recurrent_dropout = 0.2))(added)
    hidden = Dense(256, activation = "relu")(lstm)
    x = Dropout(0.5)(hidden)
    output = Dense(3, activation = "softmax")(x)
    model = Model(inputs = [input, imageFtrs], output = added)
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return model

def saveData(df):
    with open("image_features.csv", "w") as writeFile:
        df.to_csv(writeFile, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        writeFile.close()

def toList(list):
    return literal_eval(str(list))

def main():
    file = "./train_text_input_subset.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header = 0)
    XTrain = df["TOKENISED"].apply(toList).to_numpy() # CONVERT THIS TO NUMPY ARRAY OF LISTS
#    paths = df["IMG"].tolist()
    print(XTrain)
    paths = df["IMG"]
    print(paths)
#    imageFeatures = getImgReps(paths)
#    saveData(imageFeatures)
#    model = mainModel()
    YTrain = df["TXT_SNTMT"].to_numpy("int32")
    print(YTrain)
    # ORGANISE PARAMS FOR MODEL FITTING, THEY ARE NUMPY ARRAYS

if __name__ == "__main__":
    main()
