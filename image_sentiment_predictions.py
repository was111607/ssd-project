import pandas as pd
import numpy as np
import re
import csv
import os
from os import path
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.applications.vgg19 import preprocess_input
from keras import regularizers
import pickle
from keras.models import Model
from network_training import loadModel, initFtrVGG
from runai import ga

def t4saVGG(mainPath): # Import to image_sentiment_creation?
    reg = regularizers.l2(0.000005) # / t4sa stated decay / 2
    input = Input(shape = (224, 224, 3))
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
        name = "fc6",
        bias_regularizer = reg,
        kernel_regularizer = reg,
        trainable = True)(flatten)
    dropout1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(4096,
        activation = "relu",
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
    optimiser = SGD(lr = 0.001, momentum = 0.9) # learning_rate decays
    gaOptimiser = ga.keras.optimizers.Optimizer(optimiser, steps = 2)
    model.compile(optimizer = gaOptimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.load_weights(path.join(mainPath, "vgg19_ft_weights.h5"), by_name = True)
    #saveModel(model, mainPath, saveName, overWrite = False)
    print(model.summary())
    return model

def saveResults(dict, mainPath):
    with open(path.join(mainPath, "image_predictions.pickle"), "wb") as writeFile:
        pickle.dump(dict, writeFile)
        writeFile.close()

def saveDataFrame(df, fname):
    with open (fname, "w") as writeFile:
        df.to_csv(writeFile, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        writeFile.close()

def getFilename(path):
    return re.search(r"(?<=/)[0-9]+-[0-9].jpg", path).group(0)

def matchMainModelInput(matchings, df):
    # PErform actions within a dataframe
    df["IMG_PREDS"] = df["IMG"].apply(getFilename).map(matchings)
    return df

def matchMainModelInputFTR(matchings, df):
    # PErform actions within a dataframe
    df["IMG_FTRS"] = df["IMG"].apply(getFilename).map(matchings)
    return df

def getImgSntmts(mainPath, testLen, modelName, isFt, batchSize = 32):
    matchings = {}
    if (isFt is True) and not (path.exists(path.join(mainPath, "models", fname + ".h5"))):
        model = t4saVGG(mainPath)
        saveModel(model, mainPath, modelName, overWrite = False)
    else:
        model = loadModel(mainPath, modelName)
    dataGen = ImageDataGenerator(preprocessing_function = preprocess_input)
    dir = path.join(mainPath, "b-t4sa", "data")
    testGen = dataGen.flow_from_directory(path.join(dir, "test"), target_size=(224, 224), batch_size = batchSize, class_mode = None, shuffle = False)
    testGen.reset()
    probs = model.predict_generator(testGen, steps = -(-testLen // batchSize), verbose = 1)
    inputOrder = testGen.filenames
    for imagePath, prob in zip(inputOrder, probs):
        fileName = re.search(r"(?<=/)[0-9]+-[0-9].jpg", imagePath).group(0)
        matchings[fileName] = prob.tolist()
    saveResults(matchings, mainPath)
    return matchings

#TEST FEATURES
def getImgFtrs(mainPath, testLen, model, isFt, batchSize = 32):
    matchings = {}
    # if (isFt is True) and not (path.exists(path.join(mainPath, "models", fname + ".h5"))):
    #     model = t4saVGG(mainPath)
    #     saveModel(model, mainPath, modelName, overWrite = False)
    # else:
    #     model = loadModel(mainPath, modelName)
    dataGen = ImageDataGenerator(preprocessing_function = preprocess_input)
    dir = path.join(mainPath, "b-t4sa", "data")
    testGen = dataGen.flow_from_directory(path.join(dir, "test"), target_size=(224, 224), batch_size = batchSize, class_mode = None, shuffle = False)
    testGen.reset()
    ftrs = model.predict_generator(testGen, steps = -(-testLen // batchSize), verbose = 1)
    inputOrder = testGen.filenames
    for imagePath, ftr in zip(inputOrder, ftrs):
        fileName = re.search(r"(?<=/)[0-9]+-[0-9].jpg", imagePath).group(0)
        matchings[fileName] = ftr.tolist()
    saveResults(matchings, mainPath)
    return matchings

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")
    pd.set_option('display.max_colwidth', -1)
    dfTest = pd.read_csv(testFile, header = 0)
    testLen = dfTest.shape[0]
    #matchings = getImgSntmts(mainPath, testLen, "img_model_st", False, batchSize = 16)
    matchings = getImgFtrs(mainPath, testLen, initFtrVGG(mainPath, "img_model_st"), False, batchSize = 16)
    updatedDf = matchMainModelInputFTR(matchings, dfTest) #matchMainModelInput(matchings, dfTest)
    saveDataFrame(updatedDf, path.join(mainPath, "b-t4sa/model_input_testing_updated_st_FTRS.csv"))

if __name__ == "__main__":
    main()
