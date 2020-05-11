"""
--------------------------
Written by William Sewell
--------------------------
Performs the model testing step.

This was performed on an external system: Compute Nodes, AWS S3 (server group).
---------------
Files Required
---------------
Trained models - HDF5 files storing the model architecture integrating the trained weights configuration.

model_input_testing.csv - Stores testing split information used by the models in a single file:
                           The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                           image path, image sentiment, whether the image and text sentiments match.

image_sntmt_probs_testing.npy - Stores predicted sentiment scores resembling probabilities for the
                                existing BT4SA testing split. Required to test decision-level models.

image_sntmt_ftrs_testing.npy - Stores predicted sentiment features for the
                               existing BT4SA testing split. Required to test feature-level models.

OPTIONAL:
Testing split image sentiment probability scores predicted by a self-trained model.

Testing split image sentiment features predicted by a self-trained model.
---------------
Files Produced
---------------
Model testing scores - Saves score objects that contain resulting metrics from testing a model
"""

import pandas as pd
import numpy as np
import pickle
import os
from os import path
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import SGD
from ast import literal_eval
from networks import dFusionModel

# Evaluates the string, which stores a list of predictions, literally to infer that it is a list type
# and converts it to a list, then being retyped as a Numpy array.
def toArray(list):
    return np.array(literal_eval(str(list)))

# Attempts to load a model using the provided filename, from the models subdirectory
def loadModel(mainPath, modelName):
    try:
        model = load_model(path.join(mainPath, "models", modelName + ".h5"))
        print(modelName)
        print(model.summary())
        return model
    except OSError:
        print("Cannot find model by " + modelName + " to load.")
        exit()

# Saves the scores object containing metrics as a result of testing models into the new_scores subdirectory
# under the provided filename.
def saveScore(score, fname):
    dirName = "NEW_scores"
    if not path.exists(dirName):
        os.mkdir(dirName)
    with open(path.join(dirName, fname + ".pickle"), "wb") as writeFile:
        pickle.dump(score, writeFile)
        print("Score saved for filename: " + fname)
        print("\n")
        writeFile.close()

# Loads a model provided its name and, if it is a text-only model, intended to perform decision-level fusion and
# accompanied by multiple inputs, the decision-level architecture is initialised using the loaded model.
# The loaded model is then tested against the Y testing split labels that are the classified tweet sentiments to compare to.
# The resulting scores are displayed and saved.
def evalModel(isDecision, mainPath, modelName, input, YTest, fusionType, scoreName):
    # Configuration for alternate external directory structure
    if isDecision is True:
        textModel = loadModel(mainPath, modelName)
        model = dFusionModel(textModel)
    else:
        model = loadModel(mainPath, modelName)
    score = model.evaluate(input, to_categorical(YTest)) # One-hot encodes classifications
                                                         # to compare against softmax results.
    print(f"The loss for {fusionType} is: {score[0]}")
    print(f"The accuracy for {fusionType} is: {score[1]}")
    saveScore(score, scoreName)

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True # Set if on external system
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Set according to GPU to use
        mainPath = awsDir
    else:
        mainPath = curDir

    # Load split data as a DataFrame
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")
    dfTest = pd.read_csv(testFile, header = 0)

    # Create a Numpy array of testing input vectors to form network inputs by retrieving
    # columns storing the vectors in the DataFrame and typesetting them to a list, which can be converted to a Numpy array.
    # The tweet sentiment classifications are initialised as numpy arrays storing integers.
    XTest = np.stack(dfTest["TOKENISED"].apply(toArray))
    YTest = dfTest["TXT_SNTMT"].to_numpy("int32")

    # Loads any relevant sentiment features and classifications to form a secondary input into a
    # fusion technique-implementing model.
    testImgFtrs = np.load(path.join(mainPath, "b-t4sa/image sentiment features/image_sntmt_features_testing.npy"))
    testImgStFtrs = np.load(path.join(mainPath, "b-t4sa/image sentiment features/image_sntmt_features_testing_st.npy"))
    testImgProbs = np.load(path.join(mainPath, "b-t4sa/image sentiment classifications/image_sntmt_probs_testing.npy"))
    testImgStProbs = np.load(path.join(mainPath, "b-t4sa/image sentiment classifications/image_sntmt_probs_testing_st.npy"))

    # Model testing function calls
    ### Testing image features and probabilities predicted by VGG-T4SA FT-F and self-trained models ###
    # Original arbitrary models have not been tested

    # Base arbitrary text model
    # evalModel(False, mainPath, "textArb_model", XTest, YTest, "text only", "textArb_NonST_model_score")
    #
    # Arbitrary decision-level models
    # evalModel(True, mainPath, "textArb_model", [XTest, testImgProbs], YTest, "decision-level fusion (Non-ST images)", "decisionArbNST_model_score")
    # evalModel(True, mainPath, "textArb_model", [XTest, testImgStProbs], YTest, "decision-level fusion (ST images)", "decisionArbST_model_score")
    #
    # Arbitrary feature-level models
    # evalModel(False, mainPath, "featureArb_model", [XTest, testImgFtrs], YTest, "feature-level fusion (Non-ST images)", "featureArbNST_model_score")
    # evalModel(False, mainPath, "featureArb_model", [XTest, testImgStFtrs], YTest, "feature-level fusion (ST images)", "featureArbNST_model_score")
    #
    # Self decision-level models
    # evalModel(True, mainPath, "textSelf_model", [XTest, testImgProbs], YTest, "decision-level fusion (Non-ST images)", "decisionSelfNST_model_score")
    # evalModel(True, mainPath, "textSelf_model", [XTest, testImgStProbs], YTest, "decision-level fusion (ST images)", "decisionSelfST_model_score")
    #
    # Self feature-level models
    # evalModel(False, mainPath, "featureSelf_model", [XTest, testImgFtrs], YTest, "feature-level fusion (Non-ST images)", "featureSelfNST_model_score")
    # evalModel(False, mainPath, "featureSelf_model", [XTest, testImgStFtrs], YTest, "feature-level fusion (ST images)", "featureSelfST_model_score")


    ### Testing differing hyperparameter configurations for each model type using ST images. ###
    # Arbitrary scores can be used from the ST images tests.

    # GS Optimised models
    # evalModel(False, mainPath, "textOptimised_model", XTest, YTest, "no fusion (text only) optimised", "textOpt_model_score")
    # evalModel(True, mainPath, "textOptimised_model", [XTest, testImgStProbs], YTest, "decision-level fusion optimised (ST)", "decisionOpt_model_ST_score")
    # evalModel(False, mainPath, "featureOptimised_model", [XTest, testImgStFtrs], YTest, "feature-level fusion optimised (ST)", "featureOpt_model_ST_score")

    # Self-improved models
    evalModel(False, mainPath, "textSelf_model", XTest, YTest, "no fusion (text only) (self-improved)", "textSelf_model_score")
    evalModel(True, mainPath, "textSelf_model", [XTest, testImgStProbs], YTest, "decision-level fusion (self-improved)", "decisionSelf_model_ST_score")
    # evalModel(False, mainPath, "featureSelf_model", [XTest, testImgStFtrs], YTest, "feature-level fusion (self-improved)", "featureSelf_model_ST_score")

    # # Adam models
    # evalModel(False, mainPath, "textAdam_model", XTest, YTest, "no fusion (text only) Adam", "textAdam_model_score")
    # evalModel(True, mainPath, "textAdam_model", [XTest, testImgStProbs], YTest, "decision-level fusion (Adam)", "decisionAdam_model_ST_score")
    # feature-level Adam has not been tested

    # LR 0001 models
    # evalModel(False, mainPath, "textLr0001_model", XTest, YTest, "no fusion (text only) (lr 0.0001)", "textLr0001_model_score")
    # evalModel(True, mainPath, "textLr0001_model", [XTest, testImgStProbs], YTest, "decision-level fusion (lr 0.0001)", "decisionLr0001_model_ST_score")
    # evalModel(False, mainPath, "featureLr0001_model", [XTest, testImgStFtrs], YTest, "feature-level fusion (lr 0.0001)", "featureLr0001_model_ST_score")

if __name__ == "__main__":
    main()
