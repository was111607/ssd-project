"""
--------------------------
Written by William Sewell
--------------------------
Performs image processing using image data stored locally to acquire predictions of
image sentiment features and classifications.

This was performed on an external system: AWS S3 (server group).
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

gen_data - 344,389 images of the existing tweets organised into their sentiment classes
           under the split that they belong to: training, training subset, validation and testing.

vgg19_ft_weights.h5 - Stores the weights for the VGG-T4SA FT-F model from T4SA, downloaded and converted
                      to a Keras-compatible format using Caffe-to-Keras Weight Converter by pierluigiferrari.

OPTIONAL:
Trained image classification models - Saved as HDF5 files to be loaded in and predict the
                                      testing split data (can be converted to predict features).
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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import SGD
from keras import regularizers
import pickle
from ast import literal_eval

# VGG-T4SA FT-F model definition that initialises according to its original training configuration
# and loads its layers' weights before returning the completed model.
def t4saVGG(mainPath):
    reg = regularizers.l2(0.000005)  # Applied to bias and kernel regularizers in trainable layers
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
    optimiser = SGD(lr = 0.001, momentum = 0.9)  # learning rate decays
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

# Stores the results into a pickle file as a backup
def backupResults(dict, mainPath, saveName):
    with open(path.join(mainPath, saveName + ".pickle"), "wb") as writeFile:
        pickle.dump(dict, writeFile)
        print("saved matchings backup")
        writeFile.close()

# Attempts to load a model using the provided filename, from the models subdirectory
def loadModel(mainPath, fname):
    try:
        modelPath = path.join(mainPath, "models", fname + ".h5")
        print(modelPath)
        model = load_model(modelPath)
        print(fname + " successfully loaded")
        return model
    except OSError:
        print("Cannot find model: " + modelPath + " to load.")
        exit()

# Saves predictions, converted to a numpy array,
def savePredictions(saveName, predictions):
    np.save(saveName, np.stack(predictions)) # Predictions are concatenated into a single Numpy array.
    print("Predictions saved")

# For a provided image path, extract just the filename by matching all characters succeeding /
# Regular expression has been stated in full to avoid matching on any trailing characters
def getFilename(path):
    return re.search(r"(?<=/)[0-9]+-[0-9].jpg", path).group(0)

# Returns a DataFrame column storing model predictions in the same index as their corresponding image,
# mapped by the filename retrieved from the image path to the key storing the prediction value in the dictionary
def matchPreds(matchings, df):
    return df["IMG"].apply(getFilename).map(matchings)

# Saves the provided model in its entirety into the models directory under a provided filename,
# with the choice to overwrite an existing model if it resides in the directory
def saveModel(model, mainPath, fname, overWrite = True):
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

# Produces a dictionary associating image predictions to their image filenames as a key
# since ImageDataGenerator might retrieve data in a different order to those expressed in the ordering
# of data in the split DataFrames.
# The model is initialised on VGG-T4SA FT-F or loaded in, converting to predict features if necessary
# An ImageDataGenerator is initialised to be able to generate data and apply Keras's VGG19 preprocessing function
# on the image data it will generate to prepare the data for model iput.
# The data is then retrieved in batches according to a given size using flow_from_directory
# within predict_generator, that predicts on image data by batches as produced by the generator
# The resultant probabilities are paired with their image filename corresponding to the order
# they were generated and returned to be correctly reordered as expressed by the split DataFrames
def imgPredict(mainPath, dataLen, split, modelName, predictSntmt, firstTime, batchSize):
    matchings = {}
    # Initialises VGG-T4SA FT-F if making first-time image predictions, converting it to
    # predict sentimental features if necessary, and saves the model.
    if firstTime is True:
        print("Initialising t4sa-vgg")
        if predictSntmt is True:
            model = t4saVGG(mainPath)
        else:
            print("Modifying model to output features")
            model = ftrConvert(mainPath, t4saVGG(mainPath))
        saveModel(model, mainPath, modelName, overWrite = False)
    # Otherwise load a provided model, converting it to predict features if required.
    else:
        if predictSntmt is True:
            model = loadModel(mainPath, modelName)
        else:
            print("Modifying model to output features")
            model = ftrConvert(mainPath, loadModel(mainPath, modelName))
            saveModel(model, mainPath, modelName + "_converted_to_features", overWrite = False)
    dataGen = ImageDataGenerator(preprocessing_function = preprocess_input)
    dir = path.join(mainPath, "b-t4sa", "gen_data")
    # Images are resized to 224x224x3 when generated to be compatible as a VGG-based model input
    # and are fed to the model in the same order that they were generated, in a determined batch size.
    # Class_mode is set to None as the generator is only intended to feed data to the model.
    gen = dataGen.flow_from_directory(path.join(dir, split), target_size=(224, 224), batch_size = batchSize, class_mode = None, shuffle = False)
    gen.reset() # Ensures the generation and model-feeding order of images is preserved
    probs = model.predict_generator(gen, steps = -(-dataLen // batchSize), verbose = 1) # Steps = number of batches to predict on
    inputOrder = gen.filenames # Acquire order of image paths that were retrieved from the generator
    # Stores associated images and predictions, by order of input,
    # as filename-prediction pairs in a dictionary where the
    # filename has been extracted out of its local path name
    for imagePath, prob in zip(inputOrder, probs):
        fileName = re.search(r"(?<=/)[0-9]+-[0-9].jpg", imagePath).group(0)
        matchings[fileName] = prob
    backupResults(matchings, mainPath, "image_predictions_backup")
    return matchings

# Loads provided split file as a DataFrame, measures its length to help produce
# the number of epochs to pass into the model fit process and acquires a
# dictionary of filename-prediction pairs, reordering the predictions to mirror
# the split's DataFrame row ordering and then saving the predictions for model training input.
def predictAndSave(dir, filePath, mainPath, saveName, split, modelName, predictSntmt, firstTime, batchSize):
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(filePath, header = 0)
    len = df.shape[0]
    matchings = imgPredict(mainPath, len, split, modelName, predictSntmt, firstTime, batchSize)
    predictions = matchPreds(matchings, df)
    savePredictions(path.join(dir, saveName), predictions)

def main():
    # Configuration for alternate external directory structure
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True # Set if on external system
    firstTime = True
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

    # Make first time predictions and results saving for image sentiments
    dir = path.join(mainPath, "b-t4sa", "image sentiment classifications_test")
    if (firstTime is True): # and (not path.exists(dir)):
        #os.makedirs(dir)
        #predictAndSave(dir, trainFile, mainPath, "image_sntmt_probs_training", "train", "bt4sa_img_model_class", True, firstTime, 16)
        firstTime = True # Model has been saved
        predictAndSave(dir, trainSubFile, mainPath, "image_sntmt_probs_training_subset", "train_subset", "bt4sa_img_model_class", True, firstTime, 16)
        #firstTime = False
        predictAndSave(dir, valFile, mainPath, "image_sntmt_probs_validation", "val", "bt4sa_img_model_class", True, firstTime, 16)
        predictAndSave(dir, testFile, mainPath, "image_sntmt_probs_testing", "test", "bt4sa_img_model_class", True, firstTime, 16)
    # else:
    #     print(dir + " already exists or is not first time, skipping first time creation")

    # Make first time predictions and results saving for image sentiment features
    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    if (firstTime is True)  and (not path.exists(dir)):
        os.makedirs(dir)
        dir = path.join(mainPath, "b-t4sa", "image sentiment features")
        predictAndSave(dir, trainFile, mainPath, "image_sntmt_features_training", "train", "bt4sa_img_model_ftrs", False, firstTime, 16)
        firstTime = False # Model has been saved
        predictAndSave(dir, trainSubFile, mainPath, "image_sntmt_features_training_subset", "train_subset", "bt4sa_img_model_ftrs", False, firstTime, 16)
        predictAndSave(dir, valFile, mainPath, "image_sntmt_features_validation", "val", "bt4sa_img_model_ftrs", False, firstTime, 16)
        predictAndSave(dir, testFile, mainPath, "image_sntmt_features_testing", "test", "bt4sa_img_model_ftrs", False, firstTime, 16)
    else:
        print(dir + " already exists or is not first time, skipping first time creation")

    ### Insert self-trained image model feature predictions here ###
    # Only the test set should be predicted on
    firstTime = False
    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    # predictAndSave(dir, testFile, mainPath, "image_sntmt_features_testing_st", "test", "bt4sa_img_model_class_st", False, firstTime, 16)

    ### Insert self-trained image model sentiment predictions ###
    # Only the test set should be predicted on and
    # the self-trained model should classify sentiments that needs to be converted to output features
    dir = path.join(mainPath, "b-t4sa", "image sentiment classifications")
    # predictAndSave(dir, testFile, mainPath, "image_sntmt_probs_testing_st", "test", "bt4sa_img_model_class_st", True, firstTime, 16)

if __name__ == "__main__":
    main()
