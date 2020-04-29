import pandas as pd
import numpy as np
import re
import csv
import os
from os import path
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import SGD
from keras import regularizers
import pickle
from keras.models import Model
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
    #print(model.summary())
    return model


def backupResults(dict, mainPath, saveName):
    with open(path.join(mainPath, saveName + ".pickle"), "wb") as writeFile:
        pickle.dump(dict, writeFile)
        writeFile.close()

def loadModel(mainPath, fname):
    try:
        modelPath = path.join(mainPath, "models", fname + ".h5")
        model = load_model(modelPath)
        print(fname + " successfully loaded")
        return model
    except OSError:
        print("Cannot find model: " + modelPath + " to load.")
        exit()

# def saveDataFrame(df, fname):
#     with open (fname, "w") as writeFile:
#         df.to_csv(writeFile, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
#         writeFile.close()

def savePredictions(predictions, saveName):
    np.save(saveName, np.stack(predictions.apply(toArray)))
    print("Predictions saved")

def getFilename(path):
    return re.search(r"(?<=/)[0-9]+-[0-9].jpg", path).group(0)

def matchPreds(matchings, df, isPreds):
    # PErform actions within a dataframe
    return df["IMG"].apply(getFilename).map(matchings)

# def matchFtrs(matchings, df):
#     # PErform actions within a dataframe
#     df["IMG_FTRS"] = df["IMG"].apply(getFilename).map(matchings)
#     return df

# Converts a model to output features instead of classifications
def ftrConvert(mainPath, imgModel):
    #    imgModel = loadModel(mainPath, modelName)
    features = Dense(512, activation = "relu")(imgModel.layers[-2].output)
    model = Model(inputs = imgModel.input, outputs = features)
    optimiser = SGD(lr = 0.001, momentum = 0.9) # learning_rate decays
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def imgPredict(mainPath, dataLen, split, modelName, predictSntmt, firstTime, batchSize):
    matchings = {}
    if firstTime is True:
        print("Initialising t4sa-vgg")
        if predictSntmt is True:
            model = t4saVGG(mainPath)
        else:
            print("Modifying model to output features")
            model = ftrConvert(mainPath, t4saVGG(mainPath))
        saveModel(model, mainPath, modelName, overWrite = True)
    else:
        model = loadModel(mainPath, modelName)
    dataGen = ImageDataGenerator(preprocessing_function = preprocess_input)
    dir = path.join(mainPath, "b-t4sa", "data")
    gen = dataGen.flow_from_directory(path.join(dir, split), target_size=(224, 224), batch_size = batchSize, class_mode = None, shuffle = False)
    gen.reset()
    probs = model.predict_generator(gen, steps = -(-dataLen // batchSize), verbose = 1)
    inputOrder = gen.filenames
    for imagePath, prob in zip(inputOrder, probs):
        fileName = re.search(r"(?<=/)[0-9]+-[0-9].jpg", imagePath).group(0)
        matchings[fileName] = prob.tolist()
    backupResults(matchings, mainPath, "image_predictions_backup")
    return matchings

def predictAndSave(dir, file, mainPath, saveName, split, modelName, predictSntmt, firstTime, batchSize):
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header = 0)
    len = df.shape[0]
    matchings = imgPredict(mainPath, len, split, modelName, predictSntmt, firstTime, batchSize)
    predictions = matchPreds(matchings, df)
    savePredictions(predictions, path.join(dir, saveName))

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    firstTime = True
    isAws = True
    predictOnline = False

    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir

    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv")
    trainSubFile = path.join(mainPath, "b-t4sa/model_input_training_subset.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")

    if firstTime is True:
        dir = path.join(mainPath, "b-t4sa", "image sentiment classifications")
        predictAndSave(dir, trainFile, mainPath, "image_sntmt_probs_training", "train", "bt4sa_img_model_class", True, firstTime, 16)
        predictAndSave(dir, trainSubFile, mainPath, "image_sntmt_probs_training_subset", "train_subset", "bt4sa_img_model_class", True, firstTime, 16)
        predictAndSave(dir, valFile, mainPath, "image_sntmt_probs_val", "val", "bt4sa_img_model_class", True, firstTime, 16)
        predictAndSave(dir, testFile, mainPath, "image_sntmt_probs_test", "test", "bt4sa_img_model_class", True, firstTime, 16)

        dir = path.join(mainPath, "b-t4sa", "image sentiment features")
        predictAndSave(dir, trainFile, mainPath, "image_sntmt_features_training", "train", "bt4sa_img_model_ftrs", False, firstTime, 16)
        predictAndSave(dir, trainSubFile, mainPath, "image_sntmt_probs_features_subset", "train_subset", "bt4sa_img_model_ftrs", False, firstTime, 16)
        predictAndSave(dir, valFile, mainPath, "image_sntmt_features_val", "val", "bt4sa_img_model_ftrs", False, firstTime, 16)
        predictAndSave(dir, testFile, mainPath, "image_sntmt_features_test", "test", "bt4sa_img_model_ftrs", False, firstTime, 16)
    else:
        predictSntmt = False

    ### Self-trained image model predictions here
    # (path.exists(path.join(mainPath, "models", modelName + ".h5")))
    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    predictAndSave(dir, testFile, "image_sntmt_features_test_st", mainPath, testLen, "test", "bt4sa_img_model_class", False, firstTime, batchSize = 16)

    # sntmtMatchings = imgPredict(mainPath, testLen, "test", "bt4sa_img_model_ftrs", False, firstTime, batchSize = 16)
    # matchings = getImgFtrs(mainPath, testLen, firstTime = False, batchSize = 16)
    # updatedDf = matchFtrs(matchings, dfTest) #matchSntmts(matchings, dfTest)
    #saveDataFrame(updatedDf, path.join(mainPath, "b-t4sa/model_input_testing_updated_st_FTRS.csv"))

if __name__ == "__main__":
    main()
