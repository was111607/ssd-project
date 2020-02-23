import os
import csv
import re
import string
import pandas as pd
import pickle
import numpy as np
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Dropout, Reshape
from keras.layers.merge import add, concatenate
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.utils import to_categorical
from ast import literal_eval
from io import BytesIO
from urllib.request import urlopen
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

# Skimage to retrieve and resize from tweet links
# Maybe have to run all programs in succession to be able to run?
counter = 0

def loadImage(path):
    with urlopen(path) as url:
        img = load_img(BytesIO(url.read()), target_size=(224, 224))
    return img

def getImageRep(path):
    global counter
    print(counter)
    counter += 1
    img = loadImage(path)
    #img = load_img(str(path), target_size = (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    return img

def getImageClass(df):
    vgg19 = VGG19(weights = "imagenet")
    model = Sequential()
    for layer in vgg19.layers: # Output of FC2 layer
        model.add(layer)
    model.add(Dense(512, activation = "relu"))
#    df = df.sample(n = 10)
    df["REPRESENTATION"] = df.apply(getImageRep)
    classMatrix = np.concatenate(df["REPRESENTATION"].to_numpy()) # new with df
    #print(classMatrix.shape)
    return model.predict(classMatrix, batch_size = 64)

def getImageReps(df): # pathList old arg
#    images = []
    vgg19 = VGG19(weights = "imagenet")
    model = Sequential()
    for layer in vgg19.layers[:-1]: # Output of FC2 layer
        model.add(layer)
    model.add(Dense(512, activation = "relu"))
#    df = df.sample(n = 10)
    df["REPRESENTATION"] = df.apply(getImageRep)
    featureMatrix = np.concatenate(df["REPRESENTATION"].to_numpy()) # new with df
    #print(featureMatrix.shape)
    return model.predict(featureMatrix, batch_size = 64)

## FIGURE OUT HOW TO GET COMPUTING NODE TO PROCESS AND RETRIEVE IMAGES - TWEEPY?
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
    #return df.to_numpy()

# numarray = np.array([np.arange(1, 513), np.arange(1, 513)])
# vgg19 = VGG19(weights='imagenet')
# # model = Sequential()
# # for layer in vgg19.layers[:-1]:
# #     model.add(layer)
# reduceImgFtrs = Dense(512, activation = "relu")(vgg19.layers[-2].output)
# textFtrs = Input(shape=(512,))
# added = add([reduceImgFtrs, textFtrs])
# model = Model(inputs=[vgg19.input, textFtrs], output=added)

# Features accounted for separately
def decisionModel():
    with open("./training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    #textFtrs = Dense(embedDim, use_bias = False)(textFtrs)
    #print(textFtrs.output)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.2, recurrent_dropout = 0.2))(textFtrs)
    lstmShape = lstm.shape
    imageFtrs = Input(shape=(embedDim,)) # embedDim
    #reshapeImgFtrs = Reshape((int(lstmShape[0]), 1, embedDim))(imageFtrs)
    concat = concatenate([lstm, imageFtrs], axis = -1)
    hidden1 = Dense(756, activation = "relu")(concat) # Make similar to feature??
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1) # Make similar to feature??
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    print(model.summary())
    return model

def featureModel():
    with open("./training_counter.pickle", "rb") as readFile:
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
    model = Model(inputs = [input, imageFtrs], output = output)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    print(model.summary())
    return model

def saveData(list, fname):
    with open(fname, "w") as writeFile:
        writer = csv.writer(writeFile, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        for i in list:
            writer.writerow(i)
        writeFile.close()

def toArray(list):
    return np.array(literal_eval(str(list)))

def toURL(path): # ENABLE IN PATHS DF
    return "https://b-t4sa-images.s3.eu-west-2.amazonaws.com" + re.sub("data", "", str(path))

def main():
    trainFile = "./model_input_training_subset.csv"
    valFile = "./model_input_validation_subset.csv"
    #testFile = "./model_input_testing_subset.csv"
    pd.set_option('display.max_colwidth', -1)
    dfTrain = pd.read_csv(trainFile, header = 0)
    dfVal = pd.read_csv(valFile, header = 0)
    XTrain = np.stack(dfTrain["TOKENISED"].apply(toArray)) # CONVERT THIS TO NUMPY ARRAY OF LISTS
    XVal = np.stack(dfVal["TOKENISED"].apply(toArray))
#     print(XTrain.type())
# #    paths = df["IMG"].tolist()
#     print(XTrain)

    trainPaths = dfTrain["IMG"].apply(toURL)#.to_numpy("str")
    valPaths = dfVal["IMG"].apply(toURL)#.to_numpy("str")

    # For tweepy method use above variables but converted to lists to feed into getImgReps to use get_statuses_lookup

    #print(trainPaths)
    trainImgFeatures = getImageReps(trainPaths)
    valImgFeatures = getImageReps(valPaths)
    # input(XTrain)
    # print(XTrain[0])
    # input(XVal[0].shape)
    # input(trainImgFeatures.shape)
    # input(valImgFeatures.shape)

    #LOAD SAVED DATA AND UPLOAD PICKLE
    saveData(trainImgFeatures.tolist(), "image_features_training.csv") # MODIFY VECTOR INTO LENGTHS OF 30??? TES ARRAY LENGTH IN OTEST
    saveData(valImgFeatures.tolist(), "image_features_validation.csv")
    fModel = featureModel()
    dModel = decisionModel()
    YTrain = dfTrain["TXT_SNTMT"].to_numpy("int32")
    YVal = dfVal["TXT_SNTMT"].to_numpy("int32")
    #YTrain = df.apply(lambda x: )
    results = fModel.fit([XTrain, trainImgFeatures], to_categorical(YTrain), epochs = 500, batch_size = 64, validation_data = ([XVal, valImgFeatures], to_categorical(YVal)))
    print(results)
    # Convert validation subsets to be with the fit, investigate best epoch and batch size
    # ORGANISE PARAMS FOR MODEL FITTING, THEY ARE NUMPY ARRAYS # Multiple inputs, labels and outputs

    # dfTest = pd.read_csv(testFile, header = 0)
    # XTest = dfTest["TOKENISED"].apply(toList).to_numpy()
    # YTest = dfTest["TXT_SNTMT"].to_numpy("int32")
    # testPaths = dfTest["IMG"].to_numpy("str")
    # testImgFeatures = getImgReps(testPaths)
    # saveData(testImgFeatures, "image_features_testing.csv")
if __name__ == "__main__":
    main()
