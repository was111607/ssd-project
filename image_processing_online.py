import pandas as pd
import numpy as np
import re
import os
from os import path
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import SGD
from keras import regularizers
from runai import ga
from io import BytesIO
from urllib.request import urlopen

counter = 1
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
    #print(model.summary())
    return model

# Converts a model to output features instead of classifications
def ftrConvert(mainPath, imgModel):
    #    imgModel = loadModel(mainPath, modelName)
    features = Dense(512, activation = "relu")(imgModel.layers[-2].output)
    model = Model(inputs = imgModel.input, outputs = features)
    optimiser = SGD(lr = 0.001, momentum = 0.9) # learning_rate decays
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def saveModel(model, mainPath, fname, overWrite = False):
    dir = path.join(mainPath, "models")
    if not path.exists(dir):
        os.makedirs(dir)
    filePath = path.join(dir, fname + ".h5")
    if path.exists(filePath):
        if overWrite is True:
            msg = "Saved, replacing existing file of same name"
            model.save(filePath)
        else:
            msg = "Not saved, model already exists"
    else:
        msg = "Saved"
        model.save(filePath)
    print(fname + " - " + msg)

def loadModel(mainPath, fname):
    try:
        modelPath = path.join(mainPath, "models", fname + ".h5")
        model = load_model(modelPath)
        print(fname + " successfully loaded")
        return model
    except OSError:
        print("Cannot find model: " + modelPath + " to load.")
        exit()

def toURL(path): # ENABLE IN PATHS DF
    return "https://b-t4sa-images.s3.eu-west-2.amazonaws.com" + re.sub("data", "", str(path))

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

def batchPredict(df, model, noPartitions, mainPath, backupName, predictSntmt):
    # df = df.sample(n = 20)
    if predictSntmt is True:
        updatedPartitions = np.empty((0, 3))
    else:
        updatedPartitions = np.empty((0, 512))
    partitions = np.array_split(df, noPartitions)
    for partition in partitions:
        updatedPartitions = np.concatenate((updatedPartitions, getImgPredict(partition, model)), axis = 0)
        np.save(path.join(mainPath, backupName), updatedPartitions)
        print("Saved backup")
    return updatedPartitions

def predictAndSave(dir, filePath, mainPath, modelName, noPartitions, saveName, predictSntmt, firstTime, backupName = "backup_data"):
    global counter
    df = pd.read_csv(filePath, header = 0)
    paths = df["IMG"].apply(toURL)
    if firstTime is True:
        print("Initialising t4sa-vgg")
        if predictSntmt is True:
            model = t4saVGG(mainPath)
        else:
            print("Modifying model to output features")
            model = ftrConvert(mainPath, t4saVGG(mainPath))
        saveModel(model, mainPath, modelName, overWrite = False)
    else:
        model = loadModel(mainPath, modelName)
    print("Predicting for " + saveName)
    predictions = batchPredict(paths, model, noPartitions, mainPath, backupName, predictSntmt)
    np.save(path.join(dir, saveName), predictions)
    print("Saved to " + saveName + ".npy")
    counter = 0

def recoverPredictAndSave(dir, filePath, mainPath, modelName, noPartitions, saveName, predictSntmt, firstTime, backupLoadName = "backup_data"):
    global counter
    df = pd.read_csv(filePath, header = 0)
    paths = df["IMG"].apply(toURL)
    if firstTime is True:
        print("Initialising t4sa-vgg")
        if predictSntmt is True:
            model = t4saVGG(mainPath)
        else:
            print("Modifying model to output features")
            model = ftrConvert(mainPath, t4saVGG(mainPath))
        saveModel(model, mainPath, modelName, overWrite = False)
    else:
        model = loadModel(mainPath, modelName)
    print("Predicting for " + saveName)
    backup = np.load(path.join(mainPath, backupLoadName + ".npy"))
    backupLen = backup.shape[0]
    counter = backupLen
    backupSaveName = backupLoadName + "2"
    print(f"The backup length is {counter}")
    print(backupSaveName + ".npy will only back up the data remainder")
    predictions = batchPredict(paths.tail(-backupLen), model, noPartitions, mainPath, backupSaveName, predictSntmt)
    totalData = np.concatenate((backup, predictions), axis = 0)
    np.save(path.join(dir, saveName), totalData)
    print("Saved to " + saveName + ".npy")
    counter = 0

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True
    firstTime = True
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir

    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv")
    trainSubFile = path.join(mainPath, "b-t4sa/model_input_training_subset.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")

    dir = path.join(mainPath, "b-t4sa", "online image sentiment scores")
    if (firstTime is True) and (not path.exists(dir)):
        os.makedirs(dir) # bt4sa_img_model_class
        predictAndSave(dir, trainFile, mainPath, "bt4sa_img_model_class", 30, "image_sntmt_probs_training", True, firstTime)
        predictAndSave(dir, trainSubFile, mainPath, "bt4sa_img_model_class", 15, "image_sntmt_probs_training_subset", True, firstTime)
        predictAndSave(dir, valFile, mainPath, "bt4sa_img_model_class", 10, "image_sntmt_probs_validation", True, firstTime)
        predictAndSave(dir, testFile, mainPath, "bt4sa_img_model_class", 10, "image_sntmt_probs_testing", True, firstTime)
    else:
        print(dir + " already exists, exiting")
        exit()

    dir = path.join(mainPath, "b-t4sa", "online image sentiment features")
    if (firstTime is True) and (not path.exists(dir)):
        os.makedirs(dir) # bt4sa_img_model_ftrs
        predictAndSave(dir, trainFile, mainPath, "bt4sa_img_model_ftrs", 30, "image_sntmt_ftrs_training", False, firstTime)
        predictAndSave(dir, trainSubFile, mainPath, "bt4sa_img_model_ftrs", 15, "image_sntmt_probs_training_subset", False, firstTime)
        predictAndSave(dir, valFile, mainPath, "bt4sa_img_model_ftrs", 10, "image_sntmt_ftrs_validation", False, firstTime)
        predictAndSave(dir, testFile, mainPath, "bt4sa_img_model_ftrs", 10, "image_sntmt_ftrs_testing", False, firstTime)
    else:
        print(dir + " already exists, exiting")
        exit()

if __name__ == "__main__":
    main()
