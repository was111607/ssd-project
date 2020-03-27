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
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Dropout, RepeatVector
from keras.layers.merge import add, concatenate
from keras.applications.vgg19 import VGG19
from keras.utils import to_categorical, plot_model
from ast import literal_eval
from io import BytesIO
from urllib.request import urlopen
#from keras.wrappers.scikit_learn import KerasClassifier # for grid search
import keras.wrappers.scikit_learn as sl
from sklearn.model_selection import GridSearchCV
import types
import copy
from keras import losses
from keras.utils.generic_utils import has_arg
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

def monkeyPatchFit():
    def fit(self, x, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        # Returns
            history : object
                details about the training history at each epoch.
        """
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        if (losses.is_categorical_crossentropy(self.model.loss) and
                len(y.shape) != 2):
            y = keras.utils.np_utils.to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)

        x0 = np.array(x[i][0] for i in range(x.shape[0]))
        x1 = np.array(x[i][1] for i in range(x.shape[0]))
        #history = self.model.fit(x, y, **fit_args)
        print(x0)
        print(x1)
        history = self.model.fit([x0, x1], y, **fit_args)
        return history
    sl.BaseWrapper.fit = fit

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
#    visualiseModel(model, "decision_vgg.png")
    return model

def initFeatureVGG():
    vgg19 = VGG19(weights = "imagenet")
    model = Sequential()
    for layer in vgg19.layers[:-1]: # Output of FC2 layer
        model.add(layer)
    model.add(Dense(512, activation = "relu"))
#    visualiseModel(model, "feature_vgg.png")
    return model

# Features accounted for separately
def visualiseModel(model, fname):
    if not path.exists(fname):
        plot_model(model, to_file=fname)

def textModel():
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
    #hidden1 = Dense(512, activation = "relu")(concat) # Make similar to feature??
    hidden = Dense(256, activation = "relu")(lstm) # Make similar to feature??
    x = Dropout(0.5)(hidden)
    output = Dense(3, activation = "softmax")(x)
    model = Model(input = input, output = output)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
#    visualiseModel(model, "text_only_model.png") ### Uncomment to visualise, requires pydot and graphviz
#    print(model.summary())
    return model

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
    imageFtrs = Input(shape=(embedDim,)) # embedDim
    concat = concatenate([lstm, imageFtrs], axis = -1)
    #hidden1 = Dense(1024, activation = "relu")(concat) # Make similar to feature??
    hidden1 = Dense(512, activation = "relu")(concat) # Make similar to feature??
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1) # Make similar to feature??
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
#    visualiseModel(model, "decision_model.png") ### Uncomment to visualise, requires pydot and graphviz
    # print(model.summary())
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
    imageFtrs = Input(shape=(embedDim,))
    repeated = RepeatVector(seqLength)(imageFtrs)
    #print(textFtrs.output)
    concat = concatenate([textFtrs, repeated], axis = -1)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.2, recurrent_dropout = 0.2))(concat)
    hidden1 = Dense(512, activation = "relu")(lstm) # Make similar to feature??
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
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

def saveHistory(fname, history):
    dir = "./model histories/"
    if not path.exists(dir):
        os.mkdir(dir)
    with open(dir + fname + ".pickle", "wb") as writeFile:
        pickle.dump(history.history, writeFile)
        writeFile.close()
    print("Saved history for " + fname)

def saveResults(fname, results):
    dir = "./grid search results/"
    if not path.exists(dir):
        os.mkdir(dir)
    with open(dir + fname + ".pickle", "wb") as writeFile:
        pickle.dump(results, writeFile)
        writeFile.close()
    print("Saved grid search results for " + fname)

def saveModel(fname, model):
    dir = "./models/"
    if not path.exists(dir):
        os.mkdir(dir)
    model.save(dir + fname + ".h5")
    print("Saved model for " + fname)

def toArray(list):
    return np.array(literal_eval(str(list)))

def toURL(path): # ENABLE IN PATHS DF
    return "https://b-t4sa-images.s3.eu-west-2.amazonaws.com" + re.sub("data", "", str(path))

def batchPredict(df, model, noPartitions):
    # df = df.sample(n = 20)
    global counter
    updatedPartitions = np.empty((0, 512))
    partitions = np.array_split(df, noPartitions)
    for partition in partitions:
        updatedPartitions = np.concatenate((updatedPartitions, getImgPredict(partition, model)), axis = 0)
        np.save("backup_data", updatedPartitions)
        print("Saved backup")
        #saveData(updatedPartitions.tolist(), "backupData.csv")
    return updatedPartitions

def predictAndSave(df, model, noPartitions, saveName):
    print("Predicting for " + saveName)
    predictions = batchPredict(df, model, noPartitions)#getImgPredict(trainPaths, featureVGG)#getImageReps(trainPaths) #batchPredict
    np.save(saveName, predictions)
    #saveData(predictions.tolist(), saveName + ".csv")
    print("Saved to " + saveName + ".npy")

def recoverPredictAndSave(df, model, noPartitions, saveName, backupName):
    global counter
    print("Predicting for " + saveName)
    backup = np.load(backupName + ".npy")
    backupLen = backup.shape[0]
    counter = backupLen
    print(f"The backup length is {counter}")
    print("backup_data.npy will only back up the data remainder")
    predictions = batchPredict(df.tail(-backupLen), model, noPartitions)#getImgPredict(trainPaths, featureVGG)#getImageReps(trainPaths) #batchPredict
    totalData = np.concatenate((backup, predictions), axis = 0)
    np.save(saveName, totalData)
    #saveData(predictions.tolist(), saveName + ".csv")
    print("Saved to " + saveName + ".npy")

def getInputArray(fname):
    inputArr = pd.read_csv(fname, header = None)
    #inputArr = inputArr.sample(n = 20) #####################
    inputArr = inputArr.to_numpy()
    return inputArr

def summariseResults(results):
    means = results.cv_results_["mean_test_score"]
    stds = results.cv_results_["std_test_score"]
    parameters = results.cv_results_["params"]
    print("Best score of %f with parameters %r" % (results.best_score_, results.best_params_))
    for mean, std, parameter in zip(means, stds, parameters):
        print("Score of %f with std of %f with parameters %r" % (mean, std, parameter))

def main():
    trainFile = "./b-t4sa/model_input_training_subset.csv"
    valFile = "./b-t4sa/model_input_validation.csv"
    testFile = "./b-t4sa/model_input_testing.csv"
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

    dir = "./b-t4sa/image features"
    #recoverPredictAndSave(trainPaths, featureVGG, 20, dir + "/image_features_training", "backup_data")
    #input("Predicting and saving feature data completed")
    if not path.exists(dir): # Currently set to
        os.mkdir(dir)
        predictAndSave(trainPaths, featureVGG, 20, dir + "/image_features_training")
        predictAndSave(valPaths, featureVGG, 6, dir + "/image_features_validation")
        predictAndSave(testPaths, featureVGG, 6, dir + "/image_features_testing")
        input("Predicting and saving feature data completed")
    trainImgFeatures = np.load(dir + "/image_features_training50.npy") # getInputArray
    valImgFeatures = np.load(dir + "/image_features_validation.npy")
    testImgFeatures = np.load(dir + "/image_features_testing.npy")
    dir = "./b-t4sa/image classifications"
    #recoverPredictAndSave(trainPaths, decisionVGG, 20, dir + "/image_classifications_training", "backup_data")
    #input("Predicting and saving classification data completed")
    if not path.exists(dir): # Currently set to
        os.mkdir(dir)
        predictAndSave(trainPaths, decisionVGG, 20, dir + "/image_classifications_training") # Remove recover, change 10 to 20, remove backupName
        predictAndSave(valPaths, decisionVGG, 6, dir + "/image_classifications_validation")
        predictAndSave(testPaths, decisionVGG, 6, dir + "/image_classifications_testing")
        input("Predicting and saving classification data completed")
    trainImgClass = np.load(dir + "/image_classifications_training50.npy")
    valImgClass = np.load(dir + "/image_classifications_validation.npy")
    testImgClass = np.load(dir + "/image_classifications_testing.npy")
    #input(testImgClass.shape)

    dir = "./logs"
    if not path.exists(dir):
        os.mkdir(dir)

    earlyStoppage = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2, verbose = 1)

    # tModel = textModel()
    # tLogger = CSVLogger(dir + "/text_log.csv", append = False, separator = ",")
    # tModelHistory = tModel.fit(XTrain, to_categorical(YTrain), validation_data = (XVal, to_categorical(YVal)), epochs = 1, batch_size = 64, callbacks = [tLogger])#, earlyStoppage])
    # saveHistory("text_model_history", tModelHistory)
    # saveModel("text_model", tModel)

    # dModel = decisionModel()
    # dLogger = CSVLogger(dir + "/decision_log.csv", append = False, separator = ",")
    # dModelHistory = dModel.fit([XTrain, trainImgClass], to_categorical(YTrain), validation_data = ([XVal, valImgClass], to_categorical(YVal)), epochs = 50, batch_size = 64, callbacks = [dLogger])#, earlyStoppage])
    # saveHistory("decision_model_history", dModelHistory)
    # saveModel("decision_model", dModel)

    batchSizes = [16, 32, 64, 128, 256]
    paramGrid = dict(batch_size = batchSizes)
    monkeyPatchFit()
    dModel = sl.KerasClassifier(build_fn = decisionModel, verbose = 1, epochs = 3)
    grid = GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = -1, cv = 3)
    # print(YTrain.shape)
    # print(XTrain.shape)
    # print(trainImgClass.shape)
    XCombined = np.array([[XTrain[i], trainImgClass[i]] for i in range(XTrain.shape[0])])
    # print(XCombined[0][0])
    # print(XCombined[0][1])
    # input()
    # print(XTrain[0])
    # print(trainImgClass[0])
    # print(yes[0][0])
    # print(yes[0][1])
    # print(np.array(yes).shape)
    results = grid.fit(XCombined, to_categorical(YTrain))
    summariseResults(results)
    saveResults("batch_sizes", results)

    # fModel = featureModel()
    # fLogger = CSVLogger(dir + "/feature_log.csv", append = False, separator = ",")
    # fModelHistory = fModel.fit([XTrain, trainImgFeatures], to_categorical(YTrain), validation_data = ([XVal, valImgFeatures], to_categorical(YVal)), epochs = 1, batch_size = 64, callbacks = [fLogger])#, earlyStoppage])
    # saveHistory("feature_model_history", fModelHistory)
    # saveModel("feature_model", fModel)

if __name__ == "__main__":
    main()
