import pandas as pd
import pickle
import numpy as np
import os
from os import path
from keras.utils import to_categorical
from ast import literal_eval
from keras.wrappers.scikit_learn import KerasClassifier # for grid search for multi-input models
#import keras.wrappers.scikit_learn
import slms_search
from sklearn import model_selection # gridSearchCV
import gs_models

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

def toArray(list):
    return np.array(literal_eval(str(list)))

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
    trainFile = path.join(mainPath, "b-t4sa/model_input_training_subset.csv")
    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    if not path.exists(dir):
        print("No image data found, please run image_processing.py")
        exit()
    trainImgFeatures = np.load(path.join(dir, "image_sntmt_features_training_subset.npy")) # getInputArray # 50 FOR TUNING
    pd.set_option('display.max_colwidth', -1)
    dfTrain = pd.read_csv(trainFile, header = 0)
    XTrain = np.stack(dfTrain["TOKENISED"].apply(toArray)) # CONVERT THIS TO NUMPY ARRAY OF LISTS
    YTrain = dfTrain["TXT_SNTMT"].to_numpy("int32")

    # optimisers = [1, 2]
    # lRates = [0.0001, 0.001]
    # paramGrid = dict(optimiserChoice = optimisers, lRate = lRates)
    # model = KerasClassifier(build_fn = gs_models.textModel_Optimiser, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_optimiser_and_lRate")

    # optimisers = [1, 2]
    # lRates = [0.0001, 0.001]
    # paramGrid = dict(optimiserChoice = optimisers, lRate = lRates)
    # model = KerasClassifier(build_fn = gs_models.ftrModel_Optimiser, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_optimiser_and_lRate")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = gs_models.textModel_lstmDropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_lstm_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = gs_models.ftrModel_lstmDropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_lstm_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = gs_models.textModel_recDropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_rec_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = gs_models.ftrModel_recDropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_rec_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = gs_models.textModel_x1Dropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_x1_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = gs_models.ftrModel_x1Dropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_x1_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = gs_models.textModel_x2Dropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_x2_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = gs_models.ftrModel_x2Dropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_x2_dropout")

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # model = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_batch_sizes")
    #
    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # model = KerasClassifier(build_fn = ftrModel, verbose = 1, epochs = 5)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_batch_sizes")
if __name__ == "__main__":
    main()
