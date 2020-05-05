import pandas as pd
import pickle
import numpy as np
import os
from os import path
from keras.models import load_model
from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
from keras.utils import to_categorical
from ast import literal_eval
import models

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

# initialise using LearningRateScheduler and add as callback to training if required
def lrScheduler(epoch, lr):
    epochStep = 4
    divStep = 10
    if (epoch % epochStep == 0) and (epoch != 0):
        return lr / divStep
    return lr

def loadModel(mainPath, fname):
    try:
        modelPath = path.join(mainPath, "models", fname + ".h5")
        model = load_model(modelPath)
        print(fname + " successfully loaded")
        return model
    except OSError:
        print("Cannot find model: " + modelPath + " to load.")
        exit()

def saveHistory(fname, history, mainPath):
    dir = path.join(mainPath, "model histories")
    if not path.exists(dir):
        os.makedirs(dir)
    with open(path.join(dir, fname + ".pickle"), "wb") as writeFile:
        pickle.dump(history.history, writeFile)
        writeFile.close()
    print("Saved history for " + fname)

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

def toArray(list):
    return np.array(literal_eval(str(list)))

def getCallbacks(scheduleLr, logDir, logName):
    earlyStoppage = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2, verbose = 1)
    logger = CSVLogger(path.join(logDir, logName + ".csv"), append = False, separator = ",")
    callbacks = [earlyStoppage, logger]
    if scheduleLr is True:
        callbacks.append(LearningRateScheduler(lrScheduler, verbose = 1))
    return callbacks

# Train a text only or decision model
def trainMainModel(model, modelName, historyName, logDir, logName, trainInput, YTrain, valInput, YVal, mainPath, scheduleLr = True, batchSize = 16, epochs = 15):
    callbacks = getCallbacks(scheduleLr, logDir, logName)
    modelHistory = model.fit(trainInput, to_categorical(YTrain), validation_data = (valInput, to_categorical(YVal)), epochs = epochs, batch_size = batchSize, callbacks = callbacks)
    saveHistory(historyName, modelHistory, mainPath)
    saveModel(model, mainPath, modelName, overWrite = True)

# Train image model to improve from bt4sa fine tune
def trainImgModel(model, modelName, historyName, logDir, logName, trainLen, valLen, mainPath, scheduleLr = True, batchSize = 16, epochs = 15):
    callbacks = getCallbacks(scheduleLr, logDir, logName)
    dataGen = ImageDataGenerator(preprocessing_function = preprocess_input)
    dir = path.join(mainPath, "b-t4sa", "data")
    trainGen = dataGen.flow_from_directory(path.join(dir, "train"), target_size=(224, 224), batch_size = batchSize)
    valGen = dataGen.flow_from_directory(path.join(dir, "val"), target_size=(224, 224), batch_size = batchSize)
    modelHistory = model.fit_generator(trainGen,
        steps_per_epoch = -(-trainLen // batchSize),
        validation_data = valGen,
        validation_steps = -(-valLen // batchSize),
        epochs = epochs,
        callbacks = callbacks)
    saveHistory(historyName, modelHistory, mainPath)
    saveModel(model, mainPath, modelName, overWrite = True)

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir

    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv")
    trainSubFile = path.join(mainPath, "b-t4sa/model_input_training_subset.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    pd.set_option('display.max_colwidth', -1)
    dfTrain = pd.read_csv(trainFile, header = 0)
    dfTrainSub = pd.read_csv(trainSubFile, header = 0)
    dfVal = pd.read_csv(valFile, header = 0)

    XTrain = np.stack(dfTrain["TOKENISED"].apply(toArray)) # CONVERT THIS TO NUMPY ARRAY OF LISTS
    XTrainSub = np.stack(dfTrainSub["TOKENISED"].apply(toArray))
    XVal = np.stack(dfVal["TOKENISED"].apply(toArray))
    YTrain = dfTrain["TXT_SNTMT"].to_numpy("int32")
    YTrainSub = dfTrainSub["TXT_SNTMT"].to_numpy("int32")
    YVal = dfVal["TXT_SNTMT"].to_numpy("int32")

    # Validation on loading from csv or npy directly.
    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    if not path.exists(dir):
        print("No image data found, please run image_processing.py")
        exit()
    # featureVGG = initFtrVGG(mainPath, "img_model_st")
    # predictAndSave(trainSubPaths, featureVGG, 15, path.join(dir, "image_sntmt_features_training_50"), mainPath, "backup_data")
    print("Loading training features")
    trainImgFeatures = np.load(path.join(dir, "image_sntmt_features_training.npy")) # getInputArray # 50 FOR TUNING
    print("Loading validation features")
    valImgFeatures = np.load(path.join(dir, "image_sntmt_features_validation.npy"))

    logDir = "./logs"
    if not path.exists(logDir):
        os.makedirs(logDir)

    # trainImgModel(models.sentimentVGG(),
    #     "bt4sa_img_model_class_st",
    #     "bt4sa_img_model_probs_st_history",
    #     logDir,
    #     "bt4sa_img_model_ftrs_st_log",
    #     dfTrain.shape[0],
    #     dfVal.shape[0],
    #     mainPath)

    trainMainModel(models.textModelArb(),
        "textArb_model",
        "textArb_model_history",
        logDir,
        "textArb_log",
        XTrain,
        YTrain,
        XVal,
        YVal,
        mainPath,)
    #
    # trainMainModel(models.ftrModelArb(),
    #     "featureArb_model",
    #     "featureArb_model_history",
    #     logDir,
    #     "featureArb_log",
    #     [XTrain, trainImgFeatures],
    #     YTrain,
    #     [XVal, valImgFeatures],
    #     YVal,
    #     mainPath)


    # trainMainModel(models.textModelOpt(),
    #     "textOptimised_model",
    #     "textOptimised_history",
    #     logDir,
    #     "textOptimised_log",
    #     XTrain,
    #     YTrain,
    #     XVal,
    #     YVal,
    #     mainPath)

    # trainMainModel(models.ftrModelOpt(),
    #     "featureOptimised_model",
    #     "featureOptimised_model_history",
    #     logDir,
    #     "featureOptimised_log",
    #     [XTrain, trainImgFeatures],
    #     YTrain,
    #     [XVal, valImgFeatures],
    #     YVal,
    #     mainPath)

    # trainMainModel(models.textModelSelf(),
    #     "textSelf_model,
    #     "textSelf_model_history",
    #     logDir,
    #     "textSelf_log",
    #     XTrain,
    #     YTrain,
    #     XVal,
    #     YVal,
    #     mainPath,
    #     scheduleLr = False)
    #
    # trainMainModel(models.ftrModelSelf(),
    #     "featureSelf_model",
    #     "featureSelf_model_history",
    #     logDir,
    #     "featureSelf_log",
    #     [XTrain, trainImgFeatures],
    #     YTrain,
    #     [XVal, valImgFeatures],
    #     YVal,
    #     mainPath)

    # trainMainModel(models.textModelAdam(),
    #     "textAdam_model",
    #     "textAdam_model_history",
    #     logDir,
    #     "textAdam_log",
    #     XTrain,
    #     YTrain,
    #     XVal,
    #     YVal,
    #     mainPath,
    #     scheduleLr = False)

    # trainMainModel(models.ftrModelAdam(),
    #     "featureAdam_model",
    #     "featureAdam_model_history",
    #     logDir,
    #     "featureAdam_log",
    #     [XTrain, trainImgFeatures],
    #     YTrain,
    #     [XVal, valImgFeatures],
    #     YVal,
    #     mainPath)

    # trainMainModel(models.textModelSelfLr0001(),
    #     "textLr0001_model,
    #     "textLr0001_model_history",
    #     logDir,
    #     "textLr0001_log",
    #     XTrain,
    #     YTrain,
    #     XVal,
    #     YVal,
    #     mainPath,
    #     scheduleLr = False)
    #
    # trainMainModel(models.ftrModelSelfLr0001(),
    #     "featureLr0001_model",
    #     "featureLr0001_model_history",
    #     logDir,
    #     #  "sntmt_ftr-lvl_log",
    #     [XTrain, trainImgFeatures],
    #     YTrain,
    #     [XVal, valImgFeatures],
    #     YVal,
    #     mainPath)
if __name__ == "__main__":
    main()
