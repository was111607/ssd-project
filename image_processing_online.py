"""
--------------------------
Written by William Sewell
--------------------------
Performs image processing using image data stored externally to acquire predictions of
image sentiment features and classifications.

This was performed on an external system: Compute Nodes, AWS S3 (privately).
---------------
Files Required
---------------
model_input_training.csv - Stores training split information used by the models in a single file:
                           The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                           image path, image sentiment, whether the image and text sentiments match.

model_input_training_subset.csv - Stores 50% of the training split information into a single file:
                                  The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                                  image path, image sentiment, whether the image and text sentiments match.

model_input_testing.csv - Stores testing split information used by the models in a single file:
                          The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                          image path, image sentiment, whether the image and text sentiments match.

model_input_validation.csv - Stores validation split information used by the models in a single file:
                             The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                             image path, image sentiment, whether the image and text sentiments match.

data - 470,586 images stored in the BT4SA data set, extracted from b-t4sa_imgs.tar and stored externally
       (AWS S3 in this instance).

vgg19_ft_weights.h5 - Stores the weights for the VGG-T4SA FT-F model from T4SA, downloaded and converted
                      to a Keras-compatible format using Caffe-to-Keras Weight Converter by pierluigiferrari.

OPTIONAL:
Trained image classification models - Saved as HDF5 files to be loaded in and predict the
                                      testing split data (can be converted to predict features)
---------------
Files Produced
---------------
image_sntmt_probs_training.npy - Stores predicted sentiment scores resembling probabilities for the
                                 existing BT4SA training split.

image_sntmt_probs_training_subset.npy - Stores predicted sentiment scores resembling probabilities for the
                                        existing BT4SA training split subset.

image_sntmt_probs_validation.npy - Stores predicted sentiment scores resembling probabilities for the
                                   existing BT4SA validation split.

image_sntmt_probs_testing.npy - Stores predicted sentiment scores resembling probabilities for the
                                existing BT4SA testing split.

image_sntmt_ftrs_training.npy - Stores predicted sentiment features for the
                                existing BT4SA training split.

image_sntmt_ftrs_training_subset.npy - Stores predicted sentiment features for the
                                       existing BT4SA training split subset.

image_sntmt_ftrs_validation.npy - Stores predicted sentiment features for the
                                  existing BT4SA validation split.

image_sntmt_ftrs_testing.npy - Stores predicted sentiment features for the
                               existing BT4SA testing split.

OPTIONAL:
Testing split image sentiment probability scores predicted by a self-trained model.

Testing split image sentiment features predicted by a self-trained model.
"""

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
from io import BytesIO
from urllib.request import urlopen

# Global counter to track number of images processed
counter = 1

# VGG-T4SA FT-F model definition that initialises according to its original training configuration
# and loads its layers' weights before returning the completed model.
def t4saVGG(mainPath):
    reg = regularizers.l2(0.000005) # Applied to bias and kernel regularizers in trainable layers
    input = Input(shape = (224, 224, 3)) # Expected input dimensions for a VGG-based architecture
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

    # Classification block
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
    optimiser = SGD(lr = 0.001, momentum = 0.9) # learning rate decays
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.load_weights(path.join(mainPath, "vgg19_ft_weights.h5"), by_name = True) # Load weights into model matching layer names
    return model

# Converts a model to output predicted features instead of classifications
# Receives the initialised image model and integrates it on top of a fully-connected layer,
# matching feature dimensions accepted by fusion technique models, and redirecting the
# initialised model's pre-output layer output into it. Setting the input layer at the loaded
# image model to process the data prior.
def ftrConvert(mainPath, imgModel):
    features = Dense(512, activation = "relu")(imgModel.layers[-2].output) # Penultimate layer output
    model = Model(inputs = imgModel.input, outputs = features) # Input is recieved by the image model
    optimiser = SGD(lr = 0.001, momentum = 0.9) # learning rate decays
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

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

# Attempts to load a model using the provided filename, from the models subdirectory
def loadModel(mainPath, fname):
    try:
        modelPath = path.join(mainPath, "models", fname + ".h5")
        model = load_model(modelPath, compile = False) # Compilation only required for training
        print(fname + " successfully loaded")
        return model
    except OSError:
        print("Cannot find model: " + modelPath + " to load.")
        exit()

# Converts an image's original local path into its equivalent AWS S3 URL to retrieve the image.
# The local 'data' directory preceeding the rest of the image path is removed and the remaining
# string is appended onto the bucket URL.
def toURL(path):
    return "https://b-t4sa-images.s3.eu-west-2.amazonaws.com" + re.sub("data", "", str(path))

# Accesses the provided AWS S3 image URL and retrieves its stored image,
# first loading it into memory from the URL using BytesIO and converting it into
# a format compatible to apply Keras's preprocessing methods to prepare the image
# for model input.
def loadImage(path):
    with urlopen(path) as url:
        img = load_img(BytesIO(url.read()), target_size=(224, 224))
    return img

# Prepares the URL-loaded image into a format acceptable by a Keras model for processing
def getImageRep(path):
    global counter
    if (counter % 100 == 0): # Tracks progress of images being processed
        print(counter)
    img = loadImage(path) # Returns image from the URL associated to the provided path
    img = img_to_array(img) # Converts image data to a Numpy array
    img = np.expand_dims(img, axis = 0) # Adds an additional axis to allow batching of image data
    img = preprocess_input(img) # Applies the same image data preprocessing techniques used in VGG
    counter += 1
    return img

# Given a DataFrame storing a split's image URLs, retrieves the corresponding prepared image data
# suitable for model input, downloaded externally and preprocessed.
def getImgReps(df):
    df["REPRESENTATION"] = df.apply(getImageRep)
    imageReps = np.concatenate(df["REPRESENTATION"].to_numpy()) # Combines image data into a single batch
                                                                # to form a model input partition
    return imageReps

# Uses the model to predict on the provided partition of image data.
def getImgPredict(df, model):
    imageReps = getImgReps(df)
    return model.predict(imageReps, batch_size = 16)

# Splits the entire provided DataFrame containing image URLs into a series of Numpy arrays
# resembling partitions that store collections of image URLs.
# The returned results of predicting a partition accumulate updatedPartitions
# That stores all predictions made for the entire split where a backup is also saved.
def batchPredict(df, model, noPartitions, mainPath, backupName, predictSntmt):
    # If image sentiments are being predicted, initialise empty Numpy array with the same dimensions
    # as the softmax output size to be able to concatenate together partitions of model predictions
    if predictSntmt is True:
        updatedPartitions = np.empty((0, 3))
    else:
        updatedPartitions = np.empty((0, 512)) # Larger dimensional empty Numpy array matching those for image sentiment features
    partitions = np.array_split(df, noPartitions)
    # Iteratively predict partitions and accumulate their results, saving the accumulated array as a backup
    for partition in partitions:
        updatedPartitions = np.concatenate((updatedPartitions, getImgPredict(partition, model)), axis = 0)
        np.save(path.join(mainPath, backupName), updatedPartitions)
        print("Saved backup to " + backupName)
    return updatedPartitions

# Loads the given data split stored at the provided path as a DataFrame and only extracts the column
# storing the image paths where they are each converted to their equivalent AWS S3 URL.
# The required model is loaded or initialised and provided to batchPredict to acquire
# predictions, saving the results to the provided path and filename.
def predictAndSave(dir, filePath, mainPath, modelName, noPartitions, saveName, predictSntmt, firstTime, backupName = "backup_data"):
    global counter
    df = pd.read_csv(filePath, header = 0)
    paths = df["IMG"].apply(toURL) # Converts paths to the equivalent AWS S3 URL.
    # Initialises VGG-T4SA FT-F if making first-time image predictions, converting it to
    # predict sentimental features if necessary, and saves the model.
    if firstTime is True:
        print("Initialising t4sa-vgg")
        if predictSntmt is True:
            model = t4saVGG(mainPath)
        else:
            print("Modifying model to output features")
            model = ftrConvert(mainPath, t4saVGG(mainPath))
        saveModel(model, mainPath, modelName)
    # Otherwise load a provided model, converting it to predict features if required.
    # First checks if model needs to be converted - only by ST models
    else:
        if (predictSntmt is False) and ("st" in saveName):
            print("Modifying model to output features")
            model = ftrConvert(mainPath, loadModel(mainPath, modelName))
        else:
            model = loadModel(mainPath, modelName)
    print("Predicting for " + saveName)
    predictions = batchPredict(paths, model, noPartitions, mainPath, backupName, predictSntmt) # acquire predictions for image URLs
    np.save(path.join(dir, saveName), predictions)
    print("Saved to " + saveName + ".npy")
    counter = 0 # Reset image processing counter for the next split

# Performs alike predictAndSave, but beginning from a provided backup file(s) instead of from scratch.
# Given the number of packups, loads and accumulates the stored predictions they store given the filename
# format. Prediction and saving continues as normal but from the URL after the tail of the backup predictions data
# As the order is determined by the overall DataFRame that is still passed in, meaning that the ordering persists.
# Loading multiple backups must be in the form: <backupName>.npy, <backupName>2.npy, <backupName>3.npy,...
def recoverPredictAndSave(dir, filePath, mainPath, modelName, noPartitions, saveName, predictSntmt, firstTime, noBackups, backupLoadName = "backup_data"):
    global counter
    df = pd.read_csv(filePath, header = 0)
    paths = df["IMG"].apply(toURL) # Converts paths to the equivalent AWS S3 URL
    # Initialises VGG-T4SA FT-F if making first-time image predictions, converting it to
    # predict sentimental features if necessary, and saves the model.
    if firstTime is True:
        print("Initialising t4sa-vgg")
        if predictSntmt is True:
            model = t4saVGG(mainPath)
        else:
            print("Modifying model to output features")
            model = ftrConvert(mainPath, t4saVGG(mainPath))
        saveModel(model, mainPath, modelName)
    # First checks if model needs to be converted - only by ST models
    # Otherwise load a provided model, converting it to predict features if required.
    else:
        if (predictSntmt is False) and ("st" in saveName):
            print("Modifying model to output features")
            model = ftrConvert(mainPath, loadModel(mainPath, modelName))
        else:
            model = loadModel(mainPath, modelName)
    print("Predicting for " + saveName)
    # Loads backups, iterating through backupnumbers to accumulate the total backup data if required
    for i in range(noBackups):
        if i == 0:
            print("Loading " + path.join(mainPath, backupLoadName + ".npy"))
            backup = np.load(path.join(mainPath, backupLoadName + ".npy"))
        else:
            curBackupNo = str(i + 1)
            print("Loading " + path.join(mainPath, backupLoadName + curBackupNo + ".npy"))
            backupPart = np.load(path.join(mainPath, backupLoadName + curBackupNo + ".npy"))
            backup = np.concatenate((backup, backupPart), axis = 0)
    backupLen = backup.shape[0] # Length of the backup array determines where to continue predictions from
    counter = backupLen
    backupSaveName = backupLoadName + str(noBackups + 1) # Dynamically assigns a save name to not overwrite current backups
    print(f"The backup length is {counter}")
    print(backupSaveName + ".npy will only back up the remaining data")
    # Predicting and saving continues as normal, passing in the rows in the URLs DataFrame after
    # the last predicted URL to continue from the last savepoint.
    predictions = batchPredict(paths.tail(-backupLen), model, noPartitions, mainPath, backupSaveName, predictSntmt)
    totalData = np.concatenate((backup, predictions), axis = 0)
    np.save(path.join(dir, saveName), totalData)
    print("Saved to " + saveName + ".npy")
    counter = 0

def main():
    # Configuration for alternate external directory structure
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = False # Set if on external system
    firstTime = False
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to GPU to use
        mainPath = awsDir
    else:
        mainPath = curDir

    # Provide splits filepaths to extract paths from to make predictions
    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv")
    trainSubFile = path.join(mainPath, "b-t4sa/model_input_training_subset.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")

    dir = path.join(mainPath, "b-t4sa", "image sentiment classifications")
    ### Insert recoverPredictAndSave methods here followed by predictAndSave calls for unpredicted data splits ###
    # recoverPredictAndSave(dir, trainFile, mainPath, "bt4sa_img_model_class", 30, "image_sntmt_probs_training", True, False, 2)
    # recoverPredictAndSave(dir, trainSubFile, mainPath, "bt4sa_img_model_class", 1, "image_sntmt_probs_training_subset", True, False, 1)

    # Make first time predictions and results saving for image sentiments
    if (firstTime is True) and (not path.exists(dir)):
        os.makedirs(dir)
        predictAndSave(dir, trainFile, mainPath, "bt4sa_img_model_class", 60, "image_sntmt_probs_training", True, firstTime)
        firstTime = False # Model has been created
        predictAndSave(dir, trainSubFile, mainPath, "bt4sa_img_model_class", 30, "image_sntmt_probs_training_subset", True, firstTime)
        predictAndSave(dir, valFile, mainPath, "bt4sa_img_model_class", 20, "image_sntmt_probs_validation", True, firstTime)
        predictAndSave(dir, testFile, mainPath, "bt4sa_img_model_class", 20, "image_sntmt_probs_testing", True, firstTime)
    else:
        print(dir + " already exists, skipping first time creation")

    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    ### Insert recoverPredictAndSave methods here followed by predictAndSave calls for unpredicted data splits ###
    # recoverPredictAndSave(dir, trainFile, mainPath, "bt4sa_img_model_ftrs", 30, "image_sntmt_features_training", False, False, 1)

    # Make first time predictions and results saving for image sentiment features
    if (firstTime is True) and (not path.exists(dir)):
        os.makedirs(dir)
        predictAndSave(dir, trainFile, mainPath, "bt4sa_img_model_ftrs", 60, "image_sntmt_features_training", False, firstTime)
        firstTime = False # Model has been created
        predictAndSave(dir, trainSubFile, mainPath, "bt4sa_img_model_ftrs", 30, "image_sntmt_features_training_subset", False, firstTime)
        predictAndSave(dir, valFile, mainPath, "bt4sa_img_model_ftrs", 20, "image_sntmt_features_validation", False, firstTime)
        predictAndSave(dir, testFile, mainPath, "bt4sa_img_model_ftrs", 20, "image_sntmt_features_testing", False, firstTime)
    else:
        print(dir + " already exists, skipping first time creation")

    firstTime = False
    ### Insert self-trained image model feature predictions here, ST MUST BE IN THE SAVENAME TO CONVERT###
    # Only the test set should be predicted on.
    # The self-trained model needs to be converted to output features.
    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    predictAndSave(dir, testFile, mainPath, "bt4sa_img_model_class_st", 20, "image_sntmt_features_testing_st", False, firstTime)

    ### Insert self-trained image model sentiment predictions here###
    # Only the test set should be predicted on.
    dir = path.join(mainPath, "b-t4sa", "image sentiment classifications")
    predictAndSave(dir, testFile, mainPath, "bt4sa_img_model_class_st", 20, "image_sntmt_probs_testing_st", True, firstTime)

if __name__ == "__main__":
    main()
