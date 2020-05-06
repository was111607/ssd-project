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
    dirName = "new_scores"
    if not path.exists(dirName):
        os.mkdir(dirName)
    with open(path.join(dirName, fname + ".pickle"), "wb") as writeFile:
        pickle.dump(score, writeFile)
        print("Score saved for filename: " + fname)
        print("\n")
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

    # ST, Non-st
    # Arb, Adam, Opt, Self, Lr0001

    # evalModel(False, mainPath, "text_lr0001", XTest, YTest, "no fusion (text only) (lr 0.0001)", "text_model_score_lr0001")
    # evalModel(False, mainPath, "text_model", XTest, YTest, "no fusion (text only) (lr 0.001)", "text_model_score_lr001")
    #evalModel(False, mainPath, "text_model_Adam", XTest, YTest, "no fusion (text only) Adam", "text_model_score_adam")
    # evalModel(True, mainPath, "text_lr0001", [XTest, testImgProbs], YTest, "decision-level fusion (lr 0.0001)", "decision_model_score_lr0001")
    # evalModel(True, mainPath, "text_lr0001", [XTest, testImgStProbs], YTest, "decision-level fusion (lr 0.001)", "decision_model_score_st_lr0001")
    # evalModel(True, mainPath, "text_model", [XTest, testImgStProbs], YTest, "decision-level fusion (lr 0.001)", "decision_model_score_st_lr001")

    #evalModel(False, mainPath, "sntmt_ftr-lvl_model_lr001", [XTest, testImgFtrs], YTest, "feature-level fusion", "sntmt_ftr-lvl_model_lr001_score")
    #evalModel(False, mainPath, "sntmt_ftr-lvl_model_lr001", [XTest, testImgFtrsCSV], YTest, "feature-level fusion (new)", "sntmt_ftr-lvl_model_lr001_flow_score")
    # evalModel(False, mainPath, "sntmt_ftr-lvl_model_adam", [XTest, testImgFtrsCSV], YTest, "feature-level fusion (Adam)", "sntmt_ftr-lvl_model_adam_score")

    # evalModel(False, mainPath, "text_lr0001", XTest, YTest, "no fusion (text only) (lr 0.0001)", "text_model_score_lr0001")

    # evalModel(True, mainPath, "textLr0001r_model", [XTest, testImgProbs], YTest, "decision-level fusion (lr 0.0001)", "decision_model_score_lr0001")
    # evalModel(True, mainPath, "textLr0001_model", [XTest, testImgStProbs], YTest, "decision-level fusion (lr 0.0001)", "decision_model_score_st_lr0001")
################################## ST, non-st - DO FOR ARB
     evalModel(False, mainPath, "textArb_model", XTest, YTest, "text-only fusion (Non-ST images)", "textArb_NonST_model_score")
     evalModel(False, mainPath, "textArb_model", XTest, YTest, "text-only fusion (ST images)", "textArb_ST_model_score")
#
    # evalModel(True, mainPath, "textArb_model", [XTest, testImgProbs], YTest, "decision-level fusion (Non-ST images)", "decisionArb_NonST_model_score")
    # evalModel(True, mainPath, "textArb_model", [XTest, testImgStProbs], YTest, "decision-level fusion (ST images)", "decisionArb_ST_model_score")
    #
    # evalModel(False, mainPath, "featureArb_model", [XTest, testImgFtrs], YTest, "feature-level fusion (Non-ST images)", "featureArb_model_NonST_score")
    # evalModel(False, mainPath, "featureArb_model", [XTest, testImgStFtrs], YTest, "feature-level fusion (ST images)", "featureArb_model_ST_score")

################################# <Arb>, Opt, Self, Adam, Lr0001

    # - ARB

    evalModel(False, mainPath, "textOptimised_model", XTest, YTest, "no fusion (text only) optimised", "text_model_opt_score")
    evalModel(True, mainPath, "textOptimised_model", [XTest, testImgStProbs], YTest, "decision-level fusion optimised (st)", "decision_model_st_opt_score")
    evalModel(False, mainPath, "featureOptimised_model", [XTest, testImgStFtrs], YTest, "feature-level fusion optimised (st)", "sntmt_ftr-lvl_model_st_opt_score")

    # evalModel(False, mainPath, "textSelf_model", XTest, YTest, "no fusion (text only) (lr 0.001)", "text_model_score_lr001")
    # evalModel(True, mainPath, "textSelf_model", [XTest, testImgStProbs], YTest, "decision-level fusion (lr 0.001)", "decision_model_score_st_lr001")
    #evalModel(False, mainPath, "featureSelf_model", [XTest, testImgStFtrs], YTest, "feature-level fusion (new)", "sntmt_ftr-lvl_model_lr001_flow_score")

    #evalModel(False, mainPath, "textAdam_model", XTest, YTest, "no fusion (text only) Adam", "text_model_score_adam")
    #evalModel(True, mainPath, "textAdam_model", [XTest, testImgStProbs], YTest, "decision-level fusion (Adam)", "decision_model_score_adam")
    ### not required - show evidence using text model evalModel(False, mainPath, "featureAdam_model", [XTest, testImgStFtrs], YTest, "feature-level fusion (Adam)", "sntmt_ftr-lvl_model_adam_score")

    # evalModel(False, mainPath, "textLr0001_model", XTest, YTest, "no fusion (text only) (lr 0.0001)", "text_model_score_lr0001")
    # evalModel(True, mainPath, "textLr0001_model", [XTest, testImgStProbs], YTest, "decision-level fusion (lr 0.0001)", "decision_model_score_st_lr0001") REPLACE WITH ARBS
    #evalModel(False, mainPath, "featureLr0001_model", [XTest, testImgStFtrs], YTest, "feature-level fusion (new)", "sntmt_ftr-lvl_model_lr001_flow_score")


if __name__ == "__main__":
    main()
