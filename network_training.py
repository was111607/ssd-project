"""
--------------------------
Written by William Sewell
--------------------------
Performs the network creation and training step, initialising networks as defined
in networks.py

This was performed on an external system: AWS S3 (server group)
---------------
Files Required
---------------
model_input_training.csv - Stores training split information used by the models in a single file:
                           The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                           image path, image sentiment, whether the image and text sentiments match.

model_input_validation.csv - Stores validation split information used by the models in a single file:
                             The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                             image path, image sentiment, whether the image and text sentiments match.


image_sntmt_ftrs_training.npy - Stores predicted sentiment features for the
                                existing BT4SA training split.

image_sntmt_ftrs_validation.npy - Stores predicted sentiment features for the
                                  existing BT4SA validation split.

vgg19_ft_weights.h5 - Stores the weights for the VGG-T4SA FT-F model from T4SA, downloaded and converted
                      to a Keras-compatible format using Caffe-to-Keras Weight Converter by pierluigiferrari.

OPTIONAL:
gen_data - 344,389 images of the existing tweets organised into their sentiment classes
           under the split that they belong to: training, training subset, validation and testing.
           REQUIRED for training an image model.

Testing split image sentiment probability scores predicted by a self-trained model.

Testing split image sentiment features predicted by a self-trained model.
---------------
Files Produced
---------------
Trained models - HDF5 files storing the model architecture integrating the trained weights configuration.

Model histories - Pickle files storing metrics for networks during training/validation.

Log files - CSVs providing training and validation losses and accuracies to back up history objects.
"""

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

# Defines the learning rate scheduler that, if applied to networks during training,
# calculates the learning rate to train at each epoch.
# This implements T4SA's learning rate decay of a factor of 10 reduction every 5 epochs
def lrScheduler(epoch, lr):
    epochStep = 4 # epoch indexes from 0
    divStep = 10
    if (epoch % epochStep == 0) and (epoch != 0):
        return lr / divStep
    return lr # returns the Currentl learning rate if not affected

# Attempts to load a model using the provided filename, from the models subdirectory
def loadModel(mainPath, fname):
    try:
        modelPath = path.join(mainPath, "models", fname + ".h5")
        model = load_model(modelPath)
        print(fname + " successfully loaded")
        return model
    except OSError:
        print("Cannot find model: " + modelPath + " to load.")
        exit()

# Saves the resulting history object produced post model training into the model histories
# subdirectory under the given filename.
# Ideal naming format is <FusionType>_model_history.pickle for readable labelling in esults visualisation.
def saveHistory(fname, history, mainPath):
    dir = path.join(mainPath, "model histories")
    if not path.exists(dir):
        os.makedirs(dir)
    with open(path.join(dir, fname + ".pickle"), "wb") as writeFile:
        pickle.dump(history.history, writeFile) # history attribute stores the relevant metrics
        writeFile.close()
    print("Saved history for " + fname)

# Saves the provided model in its entirety into the models directory under a provided filename,
# with the choice to overwrite an existing model if it resides in the directory
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
    print(fname + " - " + msg) # Outputs status of saving the model

# Evaluates the string, which stores a list of predictions, literally to infer that it is a list type
# and converts it to a list, then being retyped as a Numpy array
def toArray(list):
    return np.array(literal_eval(str(list)))

# Initialises and returns callbacks to augment the model training process with,
# With early stoppage and logging mandatory throughout and learning rate schedular is optional
def getCallbacks(scheduleLr, logDir, logName):
    # The training will stop if validation loss is increasing for 2 epochs straight
    earlyStoppage = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2, verbose = 1)
    # Logs the model training and validation accuracies and losses
    logger = CSVLogger(path.join(logDir, logName + ".csv"), append = False, separator = ",")
    callbacks = [earlyStoppage, logger] # Combine callbacks into a list that is passed
                                        # as a parameter to model training
    if scheduleLr is True:
        callbacks.append(LearningRateScheduler(lrScheduler, verbose = 1))
    return callbacks

# Trains a text-only, decision-level or feature-level fusion model which has been initialised
# and passed in as a parameter
def trainMainModel(model, modelName, historyName, logDir, logName, trainInput, YTrain, valInput, YVal, mainPath, scheduleLr = True, batchSize = 16, epochs = 15):
    # retrieves list of callbacks to pass to model.fit
    callbacks = getCallbacks(scheduleLr, logDir, logName)
    # Trains the provided model given its input, one-hot encoded Y labels to compare to and validation data
    modelHistory = model.fit(trainInput, to_categorical(YTrain), validation_data = (valInput, to_categorical(YVal)), epochs = epochs, batch_size = batchSize, callbacks = callbacks)
    saveHistory(historyName, modelHistory, mainPath) # save metrics-storing history attribute from the resulting histotry object
    saveModel(model, mainPath, modelName, overWrite = True)

# Self-train an image model to attempt improving on VGG-T4SA FT-F given an initialised image.
# model. Must use a fit_generator as data set is too large to be loaded in and used in trainMainModel.
# A data generator is created to retrieve image data from data_gen, prepared using Keras's provided function
# to create a suitable input to VGG19-based models (it is only intended to train models using this architecture, currently).
# Two flows are established to generate data from the training and validation directories, provided with the lengths to calculate
# The number of batches to produce to train and validate using all the data.
def trainImgModel(model, modelName, historyName, logDir, logName, trainLen, valLen, mainPath, scheduleLr = True, batchSize = 16, epochs = 15):
    # retrieves list of callbacks to pass to model.fit_generator
    callbacks = getCallbacks(scheduleLr, logDir, logName)
    dataGen = ImageDataGenerator(preprocessing_function = preprocess_input)
    dir = path.join(mainPath, "b-t4sa", "gen_data") # locate where the
    # Images are resized to 224x224x3 when generated to be compatible as a VGG-based model input
    trainGen = dataGen.flow_from_directory(path.join(dir, "train"), target_size=(224, 224), batch_size = batchSize)
    valGen = dataGen.flow_from_directory(path.join(dir, "val"), target_size=(224, 224), batch_size = batchSize)
    modelHistory = model.fit_generator(trainGen,
        steps_per_epoch = -(-trainLen // batchSize), # Number of training batches to process
        validation_data = valGen,
        validation_steps = -(-valLen // batchSize), # Number of validation batches to process
        epochs = epochs,
        callbacks = callbacks)
    saveHistory(historyName, modelHistory, mainPath) # save metrics-storing history attribute from the resulting histotry object
    saveModel(model, mainPath, modelName, overWrite = True)

def main():
    # Configuration for alternate external directory structure
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True # Set if on external system
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Set according to GPU to use
        mainPath = awsDir
    else:
        mainPath = curDir

    # Load split data as DataFrames
    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    pd.set_option('display.max_colwidth', -1)
    dfTrain = pd.read_csv(trainFile, header = 0)
    dfVal = pd.read_csv(valFile, header = 0)

    # Create a Numpy array of training and validation input vectors to form network inputs by retrieving
    # columns storing the vectors in the DataFrame and typesetting them to a list, which can be converted to a Numpy array.
    # The tweet sentiment classifications are initialised as numpy arrays storing integers
    XTrain = np.stack(dfTrain["TOKENISED"].apply(toArray))
    XVal = np.stack(dfVal["TOKENISED"].apply(toArray))
    YTrain = dfTrain["TXT_SNTMT"].to_numpy("int32")
    YVal = dfVal["TXT_SNTMT"].to_numpy("int32")

    # Checks if sentiment features exist and warns the user if not
    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    if not path.exists(dir):
        print("No image data found, only text models can be trained until image processing is carried out.")
        exit()

    # Loads image features ready for model input
    print("Loading training features")
    trainImgFeatures = np.load(path.join(dir, "image_sntmt_features_training.npy"))
    print("Loading validation features")
    valImgFeatures = np.load(path.join(dir, "image_sntmt_features_validation.npy"))

    # Creates directory to store logs from training
    logDir = "./logs"
    if not path.exists(logDir):
        os.makedirs(logDir)

    ### Train networks here, must initialise networks from networks.py as part of training method call

    # trainImgModel(models.sentimentVGG(),
    #     "bt4sa_img_model_class_st",
    #     "bt4sa_img_model_probs_st_history",
    #     logDir,
    #     "bt4sa_img_model_ftrs_st_log",
    #     dfTrain.shape[0],
    #     dfVal.shape[0],
    #     mainPath)

    # trainMainModel(models.textModelArb(),
    #     "textArb_model",
    #     "textArb_model_history",
    #     logDir,
    #     "textArb_log",
    #     XTrain,
    #     YTrain,
    #     XVal,
    #     YVal,
    #     mainPath)

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
    #     "textSelf_model_TEST",
    #     "textSelf_model_historyTEST",
    #     logDir,
    #     "textSelf_logTEST",
    #     XTrain,
    #     YTrain,
    #     XVal,
    #     YVal,
    #     mainPath)
    #
    trainMainModel(models.ftrModelSelf(),
        "featureSelf_model_TEST",
        "featureSelf_model_history_TEST",
        logDir,
        "featureSelf_log_TEST",
        [XTrain, trainImgFeatures],
        YTrain,
        [XVal, valImgFeatures],
        YVal,
        mainPath)

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
    #     mainPath.
    #     scheduleLr = False)

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
    #     mainPath,
    #     scheduleLr = False)
if __name__ == "__main__":
    main()
