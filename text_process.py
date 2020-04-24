import csv
import re
import pandas as pd
import pickle
import numpy as np
import os
from os import path
from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler
from keras.models import Model, Sequential, load_model, model_from_json
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import Dense, Embedding, LSTM, Input, Lambda, Bidirectional, Flatten, Dropout, RepeatVector, Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import to_categorical, plot_model
from keras import regularizers
from keras.optimizers import SGD
from ast import literal_eval
from io import BytesIO
from urllib.request import urlopen
#from keras.wrappers.scikit_learn import KerasClassifier # for grid search for multi-input models
import keras.wrappers.scikit_learn
#import sklearn.model_selection
from slms_search import GridSearchCV
#from sklearn.model_selection import GridSearchCV
import types
import copy
from keras import losses
from keras.utils.generic_utils import has_arg
from keras import backend as K
import traceback
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
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    counter += 1
    return img

def getImgReps(df):
    df["REPRESENTATION"] = df.apply(getImageRep)
    imageReps = np.concatenate(df["REPRESENTATION"].to_numpy()) # new with df
    return imageReps

def getImgPredict(df, model): # pathList old arg
    imageReps = getImgReps(df)
    return model.predict(imageReps, batch_size = 16)

# initialise using LearningRateScheduler and add as callback to training if required
def scheduledLr(epoch):
    initialLr = 0.001
    if epoch % 4 == 0:
        decayStep = epoch // 4
        return initialLr / (10 ** decayStep)

def t4saVGG(mainPath): # evaluate gen
#    vgg19 = VGG19(weights = None, include_top = False, input_tensor = input)
    layerNames = ["conv1_1",
        "conv1_2",
        "conv2_1",
        "conv2_2",
        "conv3_1",
        "conv3_2",
        "conv3_3",
        "conv3_4",
        "conv4_1",
        "conv4_2",
        "conv4_3",
        "conv4_4",
        "conv5_1",
        "conv5_2",
        "conv5_3",
        "conv5_4"]
    layerCounter = 0
    # for layer in vgg19.layers:
    #     model.add(layer)
    # vgg19out = vgg19.output
    reg = regularizers.l2(0.000005) # / t4sa stated decay / 2
    input = Input(shape = (224, 224, 3))
    # Block 1
    x = Conv2D(64, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv1_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(input)
    x = Conv2D(64, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv1_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv2_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(128, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv2_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv3_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(256, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv3_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(256, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv3_3",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(256, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv3_4",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block3_pool")(x)

    # Block 4
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv4_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv4_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv4_3",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv4_4",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv5_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv5_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv5_3",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv5_4",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = False)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block5_pool")(x)
    flatten = Flatten(name = "flatten")(x)
    hidden1 = Dense(4096,
        activation = "relu",
        padding = "same",
        name = "fc6",
        bias_regularizer = reg,
        kernel_regularizer = reg,
        trainable = True)(flatten)
    dropout1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(4096,
        activation = "relu",
        padding = "same",
        name = "fc7",
        bias_regularizer = reg,
        kernel_regularizer = reg,
        trainable = True)(dropout1)
    dropout2 = Dropout(0.5)(hidden2)
    output = Dense(3,
        activation = "softmax",
        name = "fc8-retrain",
        bias_regularizer = reg,
        kernel_regularizer = reg,
        trainable = True)(dropout2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.0, momentum = 0.9) # learning_rate decays
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    for layer in model.layers:
        print(layer.name)
    # for layer in model.layers:
    #     if "conv" in layer.name:
    #         print("set to" + layerNames[layerCounter])
    #         layer.name = layerNames[layerCounter]
    #         layerCounter += 1
    #     for attribute in ["kernel_regularizer", "bias_regularizer"]:
    #         if (hasattr(layer, attribute) is True) and (layer.trainable is True):
    #             print("regs set")
    #             setattr(layer, attribute, regulariser)
    # print("before:")
    # for layer in model.layers:
    #     print(layer.weights)
    #     print("\n")
    model.load_weights(path.join(mainPath, "vgg19_ft_weights.h5"), by_name = True)
    print("after:")
    for layer in model.layers:
        print(layer.weights)
        print("\n")
    print(model.summary())
    saveModel(model, mainPath, "vgg19_ft")
    # try:
    #     dir = path.join(mainPath, "VGG_ft_structure.json")
    #     modelJson = model.to_json()
    #     with open(dir, "w") as writeJson:
    #         writeJson.write(modelJson)
    #         writeJson.close()
    #     # Reload json to implement change in regularizers
    #     # model.save_weights("yes.h5")
    #     with open(dir, "r") as readJson:
    #         modelJson = readJson.read()
    #         readJson.close()
    # except Exception as e:
    #     print(traceback.format_exc())
    #     exit()
    # model = model_from_json(modelJson)
    model = loadModel(mainPath, "vgg19_ft")
    model.load_weights(path.join(mainPath, "vgg19_ft_weights.h5"), by_name = True)
    for layer in model.layers[-2]:
        layer.trainable = False
    for layer in model.layers:
        print(layer.name)
        print(layer.losses)
        print(layer.weights)
        print("\n")
    input()
    return model

def sentimentVGG():
    vgg19 = VGG19(weights = "imagenet")
    model = Sequential()
    for layer in vgg19.layers[:-1]:
        model.add(layer)
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = "softmax"))
    # for layer in model.layers[:-8]:
    #     layer.trainable = False
    model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def categoryVGG():
    vgg19 = VGG19(weights = "imagenet")
    model = Sequential()
    for layer in vgg19.layers:
        model.add(layer)
    model.add(Dense(512, activation = "relu"))
#    visualiseModel(model, "decision_vgg.png")
    return model

def featureVGG():
    vgg19 = VGG19(weights = "imagenet")
    model = Sequential()
    for layer in vgg19.layers[:-1]: # Output of FC2 layer
        model.add(layer)
    model.add(Dense(512, activation = "relu"))
#    visualiseModel(model, "feature_vgg.png")
    return model

def loadModel(mainPath, fname):
    try:
        modelPath = path.join(mainPath, "models", fname + ".h5")
        model = load_model(modelPath)
        return model
    except OSError:
        print("Cannot find model: " + modelPath + " to load.")
        exit()

# Features accounted for separately
def visualiseModel(model, fname):
    if not path.exists(fname):
        plot_model(model, to_file=fname)

def textModel():# (dRate = 0.0): # (lr = 0.0, mom = 0.0): # (dRate = 0.0)
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
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.1, recurrent_dropout = 0.4))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm) # Make similar to feature??
    x1 = Dropout(0.6)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1) # Make similar to feature??
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.05, momentum = 0.8)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"]) # optimizer = "adam"
#    visualiseModel(model, "text_only_model.png") ### Uncomment to visualise, requires pydot and graphviz
#    print(model.summary())
    return model

def dFusionModel():# (dRate = 0.0): # (lr = 0.0, mom = 0.0): # (dRate = 0.0)
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
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.1, recurrent_dropout = 0.4))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm) # Make similar to feature??
    x1 = Dropout(0.6)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1) # Make similar to feature??
    x2 = Dropout(0.3)(hidden2)
    textClass = Dense(3, activation = "softmax")(x2)
    imageSntmts = Input(shape=(3,))
    output = Lambda(lambda inputs: (inputs[0] / 2) + (inputs[1] / 2))([textClass, imageSntmts])
    model = Model(input = [input, imageSntmts], output = output)
    optimiser = SGD(lr = 0.05, momentum = 0.8)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"]) # optimizer = "adam"
#    visualiseModel(model, "text_only_model.png") ### Uncomment to visualise, requires pydot and graphviz
#    print(model.summary())
    return model

def catFtrModel(lr, mom): #(lr = 0.0, mom = 0.0): # (dRate): # (extraHLayers)
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
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.4))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs], axis = -1)
    hidden1 = Dense(512, activation = "relu")(concat) # Make similar to feature??
    x1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1) # Make similar to feature??
    x2 = Dropout(0.3)(hidden2)
    #if extraHLayers == 1:
    # for i in range(extraHLayers):
    #     hidden3 = Dense(128, activation = "relu")(x2)
    #     x2 = Dropout(0.3)(hidden3)
    # elif extraHLayers == 2:
    #     hidden3 = Dense(128, activation = "relu")(x2)
    #     x3 = Dropout(0.3)(hidden3)
    #     hidden4 = Dense(64, activation = "relu")(x3)
    #     x2 = Dropout(0.3)(hidden4)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = SGD(lr = lr, momentum = mom) #(lr = 0.075, momentum = 0.6)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
#    visualiseModel(model, "decision_model.png") ### Uncomment to visualise, requires pydot and graphviz
    # print(model.summary())
    return model

def compFtrModel(lr, mom): #(dRate): # (dRate):
    with open("./training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    imageFtrs = Input(shape=(embedDim,))
    repeated = RepeatVector(seqLength)(imageFtrs)
    #print(textFtrs.output)
    concat = concatenate([textFtrs, repeated], axis = -1)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.8))(concat) # 0.8, 0.0
    hidden1 = Dense(512, activation = "relu")(lstm) # Make similar to feature??
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = SGD(lr = lr, momentum = mom)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    # visualiseModel(model, "feature_model.png") ### Uncomment to visualise, requires pydot and graphviz
    # print(model.summary())
    return model

def saveData(list, fname):
    with open(fname, "w") as writeFile:
        writer = csv.writer(writeFile, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        for i in list:
            writer.writerow(i)
        writeFile.close()
    print(fname + " saved")

def saveHistory(fname, history, mainPath):
    dir = path.join(mainPath, "model histories")
    if not path.exists(dir):
        os.makedirs(dir)
    with open(path.join(dir, fname + ".pickle"), "wb") as writeFile:
        pickle.dump(history.history, writeFile)
        writeFile.close()
    print("Saved history for " + fname)

def saveResults(dname, results, mainPath):
    dir = path.join(mainPath, "grid search results", dname)
    os.makedirs(dir)
    with open(path.join(dir, "results.pickle"), "wb") as writeResult, open(path.join(dir, "dict.pickle"), "wb") as writeDict, open(path.join(dir, "best_score.pickle"), "wb") as writeScore, open(path.join(dir, "best_params.pickle"), "wb") as writeParams:
        pickle.dump(results, writeResult)
        pickle.dump(results.cv_results_, writeDict)
        pickle.dump(results.best_score_, writeScore)
        pickle.dump(results.best_params_, writeParams)
        writeResult.close()
        writeDict.close()
        writeScore.close()
        writeParams.close()
    print("Saved grid search results for " + dname)

def saveModel(model, mainPath, fname):
    dir = path.join(mainPath, "models")
    if not path.exists(dir):
        os.makedirs(dir)
    model.save(path.join(dir, fname + ".h5"))
    print("Saved model for " + fname)

def toArray(list):
    return np.array(literal_eval(str(list)))

def toURL(path): # ENABLE IN PATHS DF
    return "https://b-t4sa-images.s3.eu-west-2.amazonaws.com" + re.sub("data", "", str(path))

# def batchImgReps(df, noPartitions, isAws):
#     global awsDir
#     global curDir
#     pCounter = 0
#     updatedPartitions = np.empty((0, 224, 224, 3))
#     partitions = np.array_split(df, noPartitions)
#     for partition in partitions:
#         updatedPartitions = np.concatenate((updatedPartitions, getImgReps(partition)), axis = 0)
#         if (pCounter % 150 == 0):
#             if isAws is True:
#                 dir = np.save(path.join(awsDir, "backup_data"), updatedPartitions)
#             else:
#                 dir = np.save(path.join(curDir, "backup_data"), updatedPartitions)
#             print("Saved backup")
#         #saveData(updatedPartitions.tolist(), "backupData.csv")
#         pCounter += 1
#     return updatedPartitions

def batchPredict(df, model, noPartitions, mainPath, backupName):
    # df = df.sample(n = 20)
    updatedPartitions = np.empty((0, 512))
    partitions = np.array_split(df, noPartitions)
    for partition in partitions:
        updatedPartitions = np.concatenate((updatedPartitions, getImgPredict(partition, model)), axis = 0)
        dir = np.save(path.join(mainPath, backupName), updatedPartitions)
        # np.save("backup_data", updatedPartitions)
        print("Saved backup")
        #saveData(updatedPartitions.tolist(), "backupData.csv")
    return updatedPartitions

# def imgRepsAndSave(df, noPartitions, saveName, isAws):
#     print("Predicting for " + saveName)
#     predictions = batchImgReps(df, noPartitions, isAws)
#     np.save(saveName, predictions)
#     #saveData(predictions.tolist(), saveName + ".csv")
#     print("Saved to " + saveName + ".npy")

def predictAndSave(df, model, noPartitions, saveName, mainPath, backupName):
    print("Predicting for " + saveName)
    predictions = batchPredict(df, model, noPartitions, mainPath, backupName)#getImgPredict(trainPaths, featureVGG)#getImgReps(trainPaths) #batchPredict
    np.save(saveName, predictions)
    #saveData(predictions.tolist(), saveName + ".csv")
    print("Saved to " + saveName + ".npy")

# def recoverImgRepsAndSave(df, noPartitions, saveName, backupName, backupName2, isAws):
#     global counter
#     print("Predicting for " + saveName)
#     backup = np.load(backupName + ".npy")
#     backup2 = np.load(backupName2 + ".npy")
#     backupLen = backup.shape[0] + backup2.shape[0] ##########
#     counter = backupLen ######
#     backup = np.concatenate((backup, backup2), axis = 0)
#     print(f"The backup length is {counter}")
#     print("backup will only store the data remainder")
#     predictions = batchImgReps(df.tail(-backupLen), noPartitions, isAws)
#     totalData = np.concatenate((backup, predictions), axis = 0)
#     np.save(saveName, totalData)
#     #saveData(predictions.tolist(), saveName + ".csv")
#     print("Saved to " + saveName + ".npy")

def recoverPredictAndSave(df, model, noPartitions, saveName, mainPath, backupLoadName, backupSaveName):
    global counter
    print("Predicting for " + saveName)
    backup = np.load(path.join(mainPath, backupLoadName + ".npy"))
    backupLen = backup.shape[0]
    counter = backupLen
    print(f"The backup length is {counter}")
    print(backupSaveName + ".npy will only back up the data remainder")
#    predictions = batchPredict(df.tail(-backupLen), model, noPartitions)#getImgPredict(trainPaths, featureVGG)#getImgReps(trainPaths) #batchPredict
    predictions = batchPredict(df.tail(-backupLen), model, noPartitions, mainPath, backupSaveName)#getImgPredict(trainPaths, featureVGG)#getImgReps(trainPaths) #batchPredict
    totalData = np.concatenate((backup, predictions), axis = 0)
    np.save(saveName, totalData)
    #saveData(predictions.tolist(), saveName + ".csv")
    print("Saved to " + saveName + ".npy")

def summariseResults(results):
    means = results.cv_results_["mean_test_score"]
    stds = results.cv_results_["std_test_score"]
    parameters = results.cv_results_["params"]
    print("Best score of %f with parameters %r" % (results.best_score_, results.best_params_))
    for mean, std, parameter in zip(means, stds, parameters):
        print("Score of %f with std of %f with parameters %r" % (mean, std, parameter))

def trainMainModel(model, logDir, logName, trainInput, YTrain, valInput, YVal, historyName, modelName, mainPath):
    earlyStoppage = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2, verbose = 1)
    logger = CSVLogger(path.join(logDir, logName + ".csv"), append = False, separator = ",")
    modelHistory = model.fit(trainInput, to_categorical(YTrain), validation_data = (valInput, to_categorical(YVal)), epochs = 50, batch_size = 16, callbacks = [logger, earlyStoppage])
    saveHistory(historyName, modelHistory, mainPath)
    saveModel(model, mainPath, modelName)

def imageSntmtTrain(model, modelName, historyName, logDir, mainPath, trainLen, valLen, isFt):
    batchSize = 16
    earlyStoppage = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2, verbose = 1)
    logger = CSVLogger(path.join(logDir, "image_sentiments_log.csv"), append = False, separator = ",")
    cb = [earlyStoppage, logger]
    if isFt is True:
        lrScheduler = LearningRateScheduler(scheduledLr)
        cb.append(lrScheduler)
    dataGen = ImageDataGenerator(preprocessing_function = preprocess_input)
    dir = path.join(mainPath, "b-t4sa", "data")
    trainGen = dataGen.flow_from_directory(path.join(dir, "train"), target_size=(224, 224), batch_size = batchSize)
    valGen = dataGen.flow_from_directory(path.join(dir, "val"), target_size=(224, 224), batch_size = batchSize)
    modelHistory = model.fit_generator(trainGen,
        steps_per_epoch = -(-trainLen // batchSize),
        validation_data = valGen,
        validation_steps = -(-valLen // batchSize),
        epochs = 50,
        callbacks = cb)
    # saveHistory(historyName, modelHistory)
    # saveModel(model, mainPath, modelName)

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir
    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")
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

    dir = path.join(mainPath, "b-t4sa", "image features")
    #         #recoverPredictAndSave(trainPaths, featureVGG, 20, dir + "/image_features_training", "backup_data")
    #         #input("Predicting and saving feature data completed")
    if not path.exists(dir):
        os.makedirs(dir)
        featureVGG = featureVGG()
        predictAndSave(trainPaths, featureVGG, 30, path.join(dir, "image_features_training"), mainPath, "backup_data")
        predictAndSave(valPaths, featureVGG, 10, path.join(dir, "image_features_validation"), mainPath, "backup_data")
        predictAndSave(testPaths, featureVGG, 10, path.join(dir, "image_features_testing"), mainPath, "backup_data")
        input("Predicting and saving feature data completed")
    trainImgFeatures = np.load(path.join(dir, "image_features_training.npy")) # getInputArray # 50 FOR TUNING
    # valImgFeatures = np.load(path.join(dir, "image_features_validation.npy"))
    # testImgFeatures = np.load(path.join(dir, "image_features_testing.npy"))
    dir = path.join(mainPath, "b-t4sa", "image categories")
    #         #recoverpredictOrBatchAndSave(trainPaths, decisionVGG, 20, dir + "/image_classifications_training", "backup_data")
    #         #input("Predicting and saving classification data completed")
    if not path.exists(dir):
        os.makedirs(dir)
        categoryVGG = categoryVGG()
        predictAndSave(trainPaths, categoryVGG, 30, path.join(dir, "image_categories_training"), mainPath, "backup_data") # Remove recover, change 10 to 20, remove backupNam
        predictAndSave(valPaths, categoryVGG, 10, path.join(dir, "image_categories_validation"), mainPath, "backup_data")
        predictAndSave(testPaths, categoryVGG, 10, path.join(dir, "image_categories_testing"), mainPath, "backup_data")
        input("Predicting and saving categories data completed")
    trainImgCategories = np.load(path.join(dir, "image_categories_training.npy")) # 50 FOR TUNING
    # valImgCategories = np.load(path.join(dir, "image_categories_validation.npy"))
    # testImgCategories = np.load(path.join(dir, "image_categories_testing.npy"))
    #
    logDir = "./logs"
    if not path.exists(logDir):
        os.makedirs(logDir)
    #
    imageSntmtTrain(t4saVGG(mainPath),
        "decision_model",
        "decision_model_history",
        logDir,
        mainPath,
        dfTrain.shape[0],
        dfVal.shape[0],
        True)

    # trainMainModel(textModel(),
    #     logDir,
    #     "text_log",
    #     XTrain,
    #     YTrain,
    #     XVal,
    #     YVal,
    #     "text_model_history",
    #     "text_model",
    #     mainPath)
    # # tModel = textModel()
    # # tLogger = CSVLogger(dir + "/text_log.csv", append = False, separator = ",")
    # # tModelHistory = tModel.fit(XTrain, to_categorical(YTrain), validation_data = (XVal, to_categorical(YVal)), epochs = 1, batch_size = 64, callbacks = [tLogger])#, earlyStoppage])
    # # saveHistory("text_model_history", tModelHistory)
    # # saveModel(tModel, mainPath, "text_model")
    #
    # trainMainModel(catFtrModel(),
    #     logDir,
    #     "cat_ftr-lvl_log",
    #     [XTrain, trainImgCategories],
    #     YTrain,
    #     [XVal, valImgCategories],
    #     YVal,
    #     "cat_ftr-lvl_model_history",
    #     "cat_ftr-lvl_model",
    #     mainPath)
    # # dModel = catFtrModel()
    # # dLogger = CSVLogger(logDir + "/decision_log.csv", append = False, separator = ",")
    # # dModelHistory = dModel.fit([XTrain, trainImgCategories], to_categorical(YTrain), validation_data = ([XVal, valImgCategories], to_categorical(YVal)), epochs = 50, batch_size = 16, callbacks = [dLogger])#, earlyStoppage])
    # # saveHistory("decision_model_history", dModelHistory)
    # # saveModel(dModel, mainPath, "decision_model")
    #
    # trainMainModel(compFtrModel(),
    #     logDir,
    #     "cmpt_ftr-lvl_log",
    #     [XTrain, trainImgFeatures],
    #     YTrain,
    #     [XVal, valImgFeatures],
    #     YVal,
    #     "cmp_ftr-lvl_model_history",
    #     "cmp_ftr-lvl_model",
    #     mainPath)
    # fModel = compFtrModel()
    # fLogger = CSVLogger(logDir + "/feature_log.csv", append = False, separator = ",")
    # fModelHistory = fModel.fit([XTrain, trainImgFeatures], to_categorical(YTrain), validation_data = ([XVal, valImgFeatures], to_categorical(YVal)), epochs = 1, batch_size = 64, callbacks = [fLogger])#, earlyStoppage])
    # saveHistory("feature_model_history", fModelHistory)
    # saveModel(fModel, mainPath, "feature_model")

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # tModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5)
    # grid = GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("batch_sizes", results)

    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # tModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("dropouts", results)

    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # tModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("dropouts", results)

    # lrs = [0.05]
    # moms = [0.0, 0.2, 0.4, 0.6, 0.8]
    # paramGrid = dict(lr = lrs, mom = moms)
    # tModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("dropouts_005", results)

    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # paramGrid = dict(dRate = dropout)
    # tModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("lstm_dropouts", results, isAws)

    # dropout = [0.6, 0.7, 0.8, 0.9]# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # paramGrid = dict(dRate = dropout)
    # tModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("lstm_rec_dropouts_2h", results, isAws)

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # dModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5)
    # grid = GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_batch_sizes", results, isAws)

    # hiddenLayers = [0, 1]
    # paramGrid = dict(extraHLayers = hiddenLayers)
    # dModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_extra_hidden_layers_opt4", results, isAws)

    # lrs = [0.09]
    # moms = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    # paramGrid = dict(lr = lrs, mom = moms)
    # dModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_lr_008", results, isAws)

    # dropout = [0.5, 0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # dModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_h3_dropout_2h", results, isAws)

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # fModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = compFtrModel, verbose = 1, epochs = 5)
    # grid = GridSearchCV(estimator = fModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("f_batch_sizes", results, isAws)

    # dropout = [0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # fModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = compFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = fModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgFeatures[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("f_lstm_dropouts_2h", results, isAws)


    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # paramGrid = dict(dRate = dropout)
    # fModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = compFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = fModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgFeatures[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("f_lstm_rec_dropouts_2", results, isAws)

    # lrs = [0.075]
    # moms = [0.0, 0.2, 0.4, 0.6, 0.8]
    # paramGrid = dict(lr = lrs, mom = moms)
    # fModel = keras.wrappers.scikit_learn.KerasClassifier(build_fn = compFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = GridSearchCV(estimator = fModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgFeatures[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("f_lr_0075", results, isAws)

if __name__ == "__main__":
    main()
