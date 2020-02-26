import csv
import re
import pandas as pd
import pickle
import numpy as np
import os
from os import path
from keras.callbacks import CSVLogger, EarlyStopping
from keras.models import Model, Sequential
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Dropout
from keras.layers.merge import add, concatenate
from keras.applications.vgg19 import VGG19
from keras.utils import to_categorical, plot_model
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
counter = 1

def loadImage(path):
    with urlopen(path) as url:
        img = load_img(BytesIO(url.read()), target_size=(224, 224))
    return img

def getImageRep(path):
    global counter
    if (counter % 100 == 0):
        print(counter)
    img = loadImage(path)
    #img = load_img(str(path), target_size = (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    counter += 1
    return img

def getImgPredict(df, model): # pathList old arg
    df["REPRESENTATION"] = df.apply(getImageRep)
    featureMatrix = np.concatenate(df["REPRESENTATION"].to_numpy()) # new with df
    #print(featureMatrix.shape)
    return model.predict(featureMatrix, batch_size = 64)

def initDecisionVGG():
    vgg19 = VGG19(weights = "imagenet")
    model = Sequential()
    for layer in vgg19.layers: # Output of FC2 layer
        model.add(layer)
    model.add(Dense(512, activation = "relu"))
    return model

def initFeatureVGG():
    vgg19 = VGG19(weights = "imagenet")
    model = Sequential()
    for layer in vgg19.layers[:-1]: # Output of FC2 layer
        model.add(layer)
    model.add(Dense(512, activation = "relu"))
    return model

# Features accounted for separately
def visualiseModel(model, fname):
    if not path.exists(fname):
        plot_model(model, to_file=fname)

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
    hidden1 = Dense(512, activation = "relu")(concat) # Make similar to feature??
    hidden2 = Dense(256, activation = "relu")(hidden1) # Make similar to feature??
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    # visualiseModel(model, "decision_model.png") ### Uncomment to visualise, requires pydot and graphviz
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
    # visualiseModel(model, "feature_model.png") ### Uncomment to visualise, requires pydot and graphviz
    print(model.summary())
    return model

def saveData(list, fname):
    with open(fname, "w") as writeFile:
        writer = csv.writer(writeFile, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        for i in list:
            writer.writerow(i)
        writeFile.close()

def saveHistory(fname, history):
    dir = "./model histories/"
    if not path.exists(dir):
        os.mkdir(dir)
    with open(dir + fname + ".pickle", "wb") as writeFile:
        pickle.dump(history.history, writeFile)
        writeFile.close()
    print("Saved history for " + fname)

def saveModel(fname, model):
        # serialize model to JSON
    dir = "./models/"
    if not path.exists(dir):
        os.mkdir(dir)
    model.save(dir + fname + ".h5")
    # with open(dir + fname + ".csv", "w") as writeFile:
    #     writeFile.write(model.to_json())
    #     writeFile.close()
    # model.save_weights(dir + fname + "_weights" + ".h5")
    print("Saved model for " + fname)

def toArray(list):
    return np.array(literal_eval(str(list)))

def toURL(path): # ENABLE IN PATHS DF
    return "https://b-t4sa-images.s3.eu-west-2.amazonaws.com" + re.sub("data", "", str(path))

def batchPredict(df, model, noPartitions):
    #df = df.sample(n = 20)
    updatedPartitions = np.empty((0, 512))
    partitions = np.array_split(df, noPartitions)
    for partition in partitions:
        updatedPartitions = np.concatenate((updatedPartitions, getImgPredict(partition, model)), axis = 0)
    return updatedPartitions

def predictAndSave(df, model, noPartitions, saveName):
    print("Predicting for " + saveName)
    predictions = batchPredict(df, model, noPartitions)#getImgPredict(trainPaths, featureVGG)#getImageReps(trainPaths) #batchPredict
    saveData(predictions.tolist(), saveName + ".csv") # MODIFY VECTOR INTO LENGTHS OF 30??? TES ARRAY LENGTH IN OTEST
    print("Saved to " + saveName + ".csv")

def getInputArray(fname):
    input = pd.read_csv(fname, header = None)
    return input.to_numpy()

def main():
    trainFile = "./model_input_training_subset.csv"
    valFile = "./model_input_validation_subset.csv"
    testFile = "./model_input_testing_subset.csv"
    pd.set_option('display.max_colwidth', -1)
    dfTrain = pd.read_csv(trainFile, header = 0)
    dfVal = pd.read_csv(valFile, header = 0)
    dfTest = pd.read_csv(testFile, header = 0)
    XTrain = np.stack(dfTrain["TOKENISED"].apply(toArray)) # CONVERT THIS TO NUMPY ARRAY OF LISTS
    XVal = np.stack(dfVal["TOKENISED"].apply(toArray))
    XTest = np.stack(dfTest["TOKENISED"].apply(toArray))
    YTrain = dfTrain["TXT_SNTMT"].to_numpy("int32")
    YVal = dfVal["TXT_SNTMT"].to_numpy("int32")
    YTest = dfTest["TXT_SNTMT"].to_numpy("int32")

    trainPaths = dfTrain["IMG"].apply(toURL)#.to_numpy("str")
    valPaths = dfVal["IMG"].apply(toURL)#.to_numpy("str")
    testPaths = dfTest["IMG"].apply(toURL)#.to_numpy("str")

    featureVGG = initFeatureVGG()
    decisionVGG = initDecisionVGG()

    dir = "./image features"
    #if not path.exists(dir): # Currently set to
    #    os.mkdir(dir)
    #    predictAndSave(trainPaths, featureVGG, 25, dir + "/image_features_training40")
    #    predictAndSave(valPaths, featureVGG, 8, dir + "/image_features_validation40")
    predictAndSave(testPaths, featureVGG, 4, dir + "/image_features_testing40")
    #    input("Predicting and saving feature data completed")
    #trainImgFeatures = getInputArray(dir + "/image_features_training40.csv")
    #valImgFeatures = getInputArray(dir + "/image_features_validation40.csv")
    #testImgFeatures = getInputArray(dir + "/image_features_testing40.csv")

    dir = "./image classifications"
    #if not path.exists(dir): # Currently set to
    #    os.mkdir(dir)
    #    predictAndSave(trainPaths, decisionVGG, 25, dir + "/image_classifications_training40")
    #    predictAndSave(valPaths, decisionVGG, 8, dir + "/image_classifications_validation40")
    predictAndSave(testPaths, decisionVGG, 4, dir + "/image_classifications_testing40")
    #    input("Predicting and saving classification data completed")
    # trainImgClass = getInputArray(dir + "/image_classifications_training40.csv")
    # valImgClass = getInputArray(dir + "/image_classifications_validation40.csv")
    # testImgClass = getInputArray(dir + "/image_classifications_testing40.csv")
    #
    # dir = "./logs"
    # if not path.exists(dir):
    #     os.mkdir(dir)
    #
    # earlyStoppage = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10, verbose = 1)
    # fModel = featureModel()
    # fLogger = CSVLogger(dir + "/feature_log.csv", append = False, separator = ",")
    # fModelHistory = fModel.fit([XTrain, trainImgFeatures], to_categorical(YTrain), validation_data = ([XVal, valImgFeatures], to_categorical(YVal)), epochs = 500, batch_size = 64, callbacks = [fLogger, earlyStoppage])
    # saveHistory("feature_model_history", fModelHistory)
    # saveModel("feature_model", fModel)
    # print(results)

    dModel = decisionModel()
    dLogger = CSVLogger(dir + "/decision_log.csv", append = False, separator = ",")
    dModelHistory = dModel.fit([XTrain, trainImgClass], to_categorical(YTrain), validation_data = ([XVal, valImgClass], to_categorical(YVal)), epochs = 500, batch_size = 64, callbacks = [dLogger, earlyStoppage])
    saveHistory("decision_model_history", dModelHistory)
    saveModel("decision_model", dModel)

if __name__ == "__main__":
    main()
