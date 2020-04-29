import re
import pandas as pd
import pickle
import numpy as np
import os
from os import path
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Lambda, Bidirectional, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.applications.vgg19 import VGG19
from keras.utils import to_categorical
from keras import regularizers
from keras.optimizers import SGD, Adam
from ast import literal_eval
from keras.wrappers.scikit_learn import KerasClassifier # for grid search for multi-input models
#import keras.wrappers.scikit_learn
import slms_search
from sklearn import model_selection # gridSearchCV
from network_training import predictSntmtFeatures, loadModel, scheduledLr

# initialise using LearningRateScheduler and add as callback to training if required
def scheduledLr(epoch, lr):
    epochStep = 4
    divStep = 10
    if (epoch % epochStep == 0) and (epoch != 0):
        return lr / divStep
    return lr

def textModel(optimiser):# (dRate = 0.0): # (lr = 0.0, mom = 0.0): # (dRate = 0.0)
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
#    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"]) # optimizer = "adam"
    print(model.summary())
    return model

def ftrModel(optimiser): #(lr = 0.0, mom = 0.0): # (dRate): # (extraHLayers)
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
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat) # Make similar to feature??
    x1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1) # Make similar to feature??
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    # optimiser = SGD(lr = 0.0001, momentum = 0.9) #(lr = 0.075, momentum = 0.6)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    print(model.summary())
    return model

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

def toURL(path): # ENABLE IN PATHS DF
    return "https://b-t4sa-images.s3.eu-west-2.amazonaws.com" + re.sub("data", "", str(path))

def summariseResults(results):
    means = results.cv_results_["mean_test_score"]
    stds = results.cv_results_["std_test_score"]
    parameters = results.cv_results_["params"]
    print("Best score of %f with parameters %r" % (results.best_score_, results.best_params_))
    for mean, std, parameter in zip(means, stds, parameters):
        print("Score of %f with std of %f with parameters %r" % (mean, std, parameter))

def gridSearch(isMultiInput, mainPath, params, model, input, YTrain, saveName):
    if isMultiInput is True:
        XTrain = input[0]
        imageFtrs = input[1]
        grid = slms_search.GridSearchCV(estimator = model, param_grid = params, n_jobs = 1, cv = 3)
        input = np.array([[XTrain[i], imageFtrs[i]] for i in range(XTrain.shape[0])])
    else:
        grid = model_selection.GridSearchCV(estimator = model, param_grid = params, n_jobs = 1, cv = 3)
    results = grid.fit(input, to_categorical(YTrain))
    summariseResults(results)
    saveResults(saveName, results, mainPath)

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
    pd.set_option('display.max_colwidth', -1)
    dfTrain = pd.read_csv(trainFile, header = 0)
    XTrain = np.stack(dfTrain["TOKENISED"].apply(toArray)) # CONVERT THIS TO NUMPY ARRAY OF LISTS
    YTrain = dfTrain["TXT_SNTMT"].to_numpy("int32")


    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    if not path.exists(dir):
        os.makedirs(dir)
        valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
        testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")
        dfVal = pd.read_csv(valFile, header = 0)
        dfTest = pd.read_csv(testFile, header = 0)
        trainPaths = dfTrain["IMG"].apply(toURL)#.to_numpy("str")
        valPaths = dfVal["IMG"].apply(toURL)#.to_numpy("str")
        testPaths = dfTest["IMG"].apply(toURL)#.to_numpy("str")
        predictSntmtFeatures(dir, mainPath, trainPaths, valPaths, testPaths, "img_model_st")

    trainImgFeatures = np.load(path.join(dir, "image_sntmt_features_training.npy")) # getInputArray # 50 FOR TUNING
    valImgFeatures = np.load(path.join(dir, "image_sntmt_features_validation.npy"))

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # model = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_batch_sizes")
    #
    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # model = KerasClassifier(build_fn = ftrModel, verbose = 1, epochs = 5)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_batch_sizes")

    # optimisers = [SGD(lr = 0.0001, momentum = 0.9),  Adam(learning_rate = 0.0001)]
    # paramGrid = dict(optimiser = optimisers)
    # model = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_optimiser")

    optimisers = [SGD(lr = 0.0001, momentum = 0.9),  Adam(learning_rate = 0.0001)]
    paramGrid = dict(optimiser = optimisers)
    model = KerasClassifier(build_fn = ftrModel, verbose = 1, epochs = 5)
    gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_optimiser")

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("batch_sizes", results)

    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("dropouts", results)

    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("dropouts", results)

    # lrs = [0.05]
    # moms = [0.0, 0.2, 0.4, 0.6, 0.8]
    # paramGrid = dict(lr = lrs, mom = moms)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("dropouts_005", results)

    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # paramGrid = dict(dRate = dropout)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("lstm_dropouts", results, isAws)

    # dropout = [0.6, 0.7, 0.8, 0.9]# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # paramGrid = dict(dRate = dropout)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("lstm_rec_dropouts_2h", results, isAws)

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # dModel = KerasClassifier(build_fn = ftrModel, verbose = 1, epochs = 5)
    # grid = slms_search.GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgFeatures[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_batch_sizes", results, mainPath)

    # hiddenLayers = [0, 1]
    # paramGrid = dict(extraHLayers = hiddenLayers)
    # dModel = KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = slms_search.GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_extra_hidden_layers_opt4", results, isAws)

    # lrs = [0.09]
    # moms = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    # paramGrid = dict(lr = lrs, mom = moms)
    # dModel = KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = slms_search.GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_lr_008", results, isAws)

    # dropout = [0.5, 0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # dModel = KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = slms_search.GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_h3_dropout_2h", results, isAws)


if __name__ == "__main__":
    main()
