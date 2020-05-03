import pandas as pd
import numpy as np
import pickle
import os
from os import path
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import SGD
from ast import literal_eval
from models import dFusionModel

def toArray(list):
    return np.array(literal_eval(str(list)))

def loadModel(mainPath, modelName):
    try:
        model = load_model(path.join(mainPath, "models", modelName + ".h5"))
        return model
    except OSError:
        print("Cannot find model by " + modelName + " to load.")
        exit()

def saveScore(score, fname):
    if not path.exists("scores"):
        os.mkdir("scores")
    with open(path.join("scores", fname + ".pickle"), "wb") as writeFile:
        pickle.dump(score, writeFile)
        print("Score saved for filename: " + fname)
        writeFile.close()

def evalModel(isDecision, mainPath, modelName, input, YTest, fusionType, scoreName):
    if isDecision is True:
        textModel = loadModel(mainPath, modelName)
        model = dFusionModel(textModel)
    else:
        model = loadModel(mainPath, modelName)
    score = model.evaluate(input, to_categorical(YTest))
    print(f"The loss for {fusionType} is: {score[0]}")
    print(f"The accuracy for {fusionType} is: {score[1]}")
    saveScore(score, scoreName)

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")
    dfTest = pd.read_csv(testFile, header = 0)

    XTest = np.stack(dfTest["TOKENISED"].apply(toArray))
    YTest = dfTest["TXT_SNTMT"].to_numpy("int32")
    testImgFtrs = np.load(path.join(mainPath, "b-t4sa/image sentiment features/image_sntmt_features_testing.npy"))
    testImgStFtrs = np.load(path.join(mainPath, "b-t4sa/image sentiment features/image_sntmt_features_testing_st.npy"))
    testImgProbs = np.load(path.join(mainPath, "b-t4sa/image sentiment classifications/image_sntmt_probs_testing.npy"))
    testImgStProbs = np.load(path.join(mainPath, "b-t4sa/image sentiment classifications/image_sntmt_probs_testing_st.npy"))

    #print(dModel.predict([[XTest[0]], [testImgClass[0]]]))
    # evalModel(False, mainPath, "text_lr0001", XTest, YTest, "no fusion (text only) (lr 0.0001)", "text_model_score_lr0001")
    # evalModel(False, mainPath, "text_model", XTest, YTest, "no fusion (text only) (lr 0.001)", "text_model_score_lr001")
    #evalModel(False, mainPath, "text_model_adam", XTest, YTest, "no fusion (text only) Adam", "text_model_score_adam")
    # evalModel(True, mainPath, "text_lr0001", [XTest, testImgProbs], YTest, "decision-level fusion (lr 0.0001)", "decision_model_score_lr0001")
    # evalModel(True, mainPath, "text_lr0001", [XTest, testImgStProbs], YTest, "decision-level fusion (lr 0.001)", "decision_model_score_st_lr0001")
    # evalModel(True, mainPath, "text_model", [XTest, testImgStProbs], YTest, "decision-level fusion (lr 0.001)", "decision_model_score_st_lr001")
    #evalModel(False, mainPath, "sntmt_ftr-lvl_model_lr001_", [XTest, testImgFtrs], YTest, "feature-level fusion", "sntmt_ftr-lvl_model_lr001_score")
    #evalModel(False, mainPath, "sntmt_ftr-lvl_model_lr001_", [XTest, testImgFtrsCSV], YTest, "feature-level fusion (new)", "sntmt_ftr-lvl_model_lr001_flow_score")
    # evalModel(False, mainPath, "sntmt_ftr-lvl_model_adam", [XTest, testImgFtrsCSV], YTest, "feature-level fusion (Adam)", "sntmt_ftr-lvl_model_adam_score")

    # evalModel(False, mainPath, "text_lr0001", XTest, YTest, "no fusion (text only) (lr 0.0001)", "text_model_score_lr0001")

    evalModel(False, mainPath, "text_model_optimised", XTest, YTest, "no fusion (text only) optimised", "text_model_opt_score")
    evalModel(True, mainPath, "text_model_optimised", [XTest, testImgProbs], YTest, "decision-level fusion optimised", "decision_model_opt_score")
    evalModel(False, mainPath, "sntmt_ftr-lvl_model_optimised", [XTest, testImgFtrs], YTest, "feature-level fusion optimised", "sntmt_ftr-lvl_model_opt_score")
    evalModel(True, mainPath, "text_model_optimised", [XTest, testImgStProbs], YTest, "decision-level fusion optimised (st)", "decision_model_st_opt_score")
    evalModel(False, mainPath, "sntmt_ftr-lvl_model_optimised", [XTest, testImgStFtrs], YTest, "feature-level fusion optimised (st)", "sntmt_ftr-lvl_model_st_opt_score")

if __name__ == "__main__":
    main()
