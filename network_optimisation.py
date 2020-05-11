"""
--------------------------
Written by William Sewell
--------------------------
Performs the model optimisation step initialising models from gs_models.py.

This was performed on an external system: Compute Nodes, AWS S3 (server group).
---------------
Files Required
---------------
model_input_training_subset.csv - Stores 50% of the training split information into a single file:
                                  The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                                  image path, image sentiment, whether the image and text sentiments match.

Trained grid search models - HDF5 files storing the model architecture integrating the trained weights configuration.

image_sntmt_ftrs_training_subset.npy - Stores predicted sentiment features for the
                                       existing BT4SA training split subset.

slms_search.py - Modified version of sklearn.model_selection._search that is able to grid search models with multiple inputs

multi_input_scorer.py - Modified version of sklearn.model_selection._validation that
                        stores a modified _fit_and_score method definition used by the search object
                        in slms_search.py.

---------------
Files Produced
---------------
Model grid search scores - Saves score objects and its metrics resulting from grid searching a model.
"""

import pandas as pd
import pickle
import numpy as np
import os
from os import path
from keras.utils import to_categorical
from ast import literal_eval
from keras.wrappers.scikit_learn import KerasClassifier
import slms_search # Modified GridSearchCV for multi-input models
from sklearn import model_selection # Native GridSearchCV for text models
import gs_models

# Saves grid search results object and its metrics into the "grid search results"
# subdirectory under an appropriate name.
def saveResults(dname, results, mainPath):
    dir = path.join(mainPath, "grid search results", dname)
    os.makedirs(dir)
    with open(path.join(dir, "results.pickle"), "wb") as writeResult, open(path.join(dir, "dict.pickle"), "wb") as writeDict, open(path.join(dir, "best_score.pickle"), "wb") as writeScore, open(path.join(dir, "best_params.pickle"), "wb") as writeParams:
        pickle.dump(results, writeResult) # Stores the general results object
        pickle.dump(results.cv_results_, writeDict) # Stores the cross-validation results across all configurations
        pickle.dump(results.best_score_, writeScore) # Stores the metrics for the best performing configuration
        pickle.dump(results.best_params_, writeParams) # Records the best performing parameters
        writeResult.close()
        writeDict.close()
        writeScore.close()
        writeParams.close()
    print("Saved grid search results for " + dname)

# Evaluates the string, which stores a list of predictions, literally to infer that it is a list type
# and converts it to a list, then being retyped as a Numpy array.
def toArray(list):
    return np.array(literal_eval(str(list)))

# Retrieves the cross-validation mean score and standard deviation across all configurations
# and formats it for display.
def summariseResults(results):
    means = results.cv_results_["mean_test_score"]
    stds = results.cv_results_["std_test_score"]
    parameters = results.cv_results_["params"]
    print("Best score of %f with parameters %r" % (results.best_score_, results.best_params_))
    for mean, std, parameter in zip(means, stds, parameters): # Summarises scores for each model
        print("Score of %f with std of %f with parameters %r" % (mean, std, parameter))

# Initialises the grid search object using the wrapped-model that varies by the parameters dictionary passed
# as a parameter. The original grid search object is initialised for text-only models and the adjusted
# grid search object definition from slms_search is used for fusion models, where the inputs,
# passed in within a list, are transformed  into the (n_samples, n_features) format expected by the KerasClassifier wrapper.
# The results are visualised and then saved.
def gridSearch(isMultiInput, mainPath, params, model, input, YTrain, saveName):
    if isMultiInput is True:
        XTrain = input[0]
        imageFtrs = input[1]
        # n_jobs = number of parallel grid searches to run (requires parallel GPUS), runs 3 cross-folds
        grid = slms_search.GridSearchCV(estimator = model, param_grid = params, n_jobs = 1, cv = 3)
        input = np.array([[XTrain[i], imageFtrs[i]] for i in range(XTrain.shape[0])]) # Reshapes input to be accepted by KerasClassifier
    else:
        grid = model_selection.GridSearchCV(estimator = model, param_grid = params, n_jobs = 1, cv = 3)
    results = grid.fit(input, to_categorical(YTrain))
    summariseResults(results)
    saveResults(saveName, results, mainPath)

def main():
    # Configuration for alternate external directory structure.
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True # Set if on external system
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Set according to GPU to use.
        mainPath = awsDir
    else:
        mainPath = curDir
    # Establishes paths where the input data and predicted features for the training subset are located
    trainFile = path.join(mainPath, "b-t4sa/model_input_training_subset.csv")
    dir = path.join(mainPath, "b-t4sa", "image sentiment features")

    # Checks if sentiment features exist and exits the program if not.
    if not path.exists(dir):
        print("No image data found, please run image_processing.py")
        exit()

    # Load split data as a DataFrame
    trainImgFeatures = np.load(path.join(dir, "image_sntmt_features_training_subset.npy"))
    dfTrain = pd.read_csv(trainFile, header = 0)

    # Create a Numpy array of testing input vectors to form network inputs by retrieving
    # columns storing the vectors in the DataFrame and typesetting them to a list, which can be converted to a Numpy array.
    # The tweet sentiment classifications are initialised as numpy arrays storing integers.
    XTrain = np.stack(dfTrain["TOKENISED"].apply(toArray))
    YTrain = dfTrain["TXT_SNTMT"].to_numpy("int32")

    ### Establish grid searches for each defined model ###
    # Grid must be defined as a dictionary storing hyperparameter names, passed into the model definitions as parameters,
    # corresponding to lists of their values to initialise models on.
    # The models must be wrapped to be optimised using the grid sarch object were the model definition name, epochs and batch
    # size are passed in as arguments.

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
