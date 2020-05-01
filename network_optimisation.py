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
from grid_search_models import textModel_Optimiser, ftrModel_Optimiser, textModel_lstmDropout, ftrModel_lstmDropout, textModel_recDropout, ftrModel_recDropout

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
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to CPU to use
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

    optimisers = [1, 2]
    lRates = [0.0001, 0.001]
    paramGrid = dict(optimiserChoice = optimisers, lRate = lRates)
    model = KerasClassifier(build_fn = textModel_Optimiser, verbose = 1, epochs = 5, batch_size = 16)
    gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_optimiser_and_lRate")

    # optimisers = [1, 2]
    # lRates = [0.0001, 0.001]
    # paramGrid = dict(optimiserChoice = optimisers, lRate = lRates)
    # model = KerasClassifier(build_fn = ftrModel_Optimiser, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_optimiser_and_lRate")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = textModel_lstmDropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_lstm_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = ftrModel_lstmDropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_lstm_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = textModel_recDropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_rec_dropout")

    # dropout = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # model = KerasClassifier(build_fn = ftrModel_recDropout, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_rec_dropout")

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # model = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_batch_sizes")
    #
    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # model = KerasClassifier(build_fn = ftrModel, verbose = 1, epochs = 5)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_batch_sizes")

    # optimisers = [1, 2]
    # paramGrid = dict(optimiserChoice = optimisers)
    # model = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(False, mainPath, paramGrid, model, XTrain, YTrain, "text_optimiser")

    # optimisers = [1, 2]
    # paramGrid = dict(optimiserChoice = optimisers)
    # model = KerasClassifier(build_fn = ftrModel, verbose = 1, epochs = 5, batch_size = 16)
    # gridSearch(True, mainPath, paramGrid, model, (XTrain, trainImgFeatures), YTrain, "feature_optimiser")





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
