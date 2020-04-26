import pandas as pd
import numpy as np
import pickle
import os
from os import path
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import SGD
from ast import literal_eval
from network_training import dFusionModel

def toArray(list):
    return np.array(literal_eval(str(list)))

def loadModel(mainPath, modelType, modelName):
    try:
        model = load_model(path.join(mainPath, "models", modelType, modelName + ".h5"))
        return model
    except OSError:
        print("Cannot find model by " + modelName + " to load.")
        exit()

def saveScore(score, fname):
    with open(fname + ".pickle", "wb") as writeFile:
        pickle.dump(score, writeFile)
        writeFile.close()

def evalModel(mainPath, modelType, modelName, input, YTest, fusionType, scoreName):
    model = loadModel(mainPath, modelType, modelName)
    score = model.evaluate(input, to_categorical(YTest))
    print(f"The loss and accuracy for {fusionType} fusion is: {score}")
    saveScore(score, scoreName)

def evalDecisionModel(mainPath, modelType, modelName, textModelType, textModelName, input, YTest, fusionType, scoreName):
    filePath = path.join(mainPath, "models", modelType, modelName + ".h5")
    textModel = loadModel(mainPath, textModelType, textModelName)
    model = dFusionModel(mainPath, textModel)
    score = model.evaluate(input, to_categorical(YTest))
    print(f"The loss and accuracy for {fusionType} is: {score}")
    saveScore(score, scoreName)

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir
    testFile = path.join(mainPath, "b-t4sa/model_input_testing_updated_st.csv")
    dfTest = pd.read_csv(testFile, header = 0)
    XTest = np.stack(dfTest["TOKENISED"].apply(toArray))
    YTest = dfTest["TXT_SNTMT"].to_numpy("int32")
    testImgFeatures = np.load(path.join(mainPath, "b-t4sa/image features/image_features_testing.npy"))
    testImgCategories = np.load(path.join(mainPath, "b-t4sa/image categories/image_categories_testing.npy"))
    if "IMG_PREDS" in dfTest.columns:
        testImgSntmtProbs = np.stack(dfTest["IMG_PREDS"].apply(toArray))
    # fModel = loadModel("training_all", "feature_model")
    # dModel = loadModel("training_all", "decision_model")
    #tModel = loadModel("text_model")

    #print(dModel.predict([[XTest[0]], [testImgClass[0]]]))
    evalModel(mainPath, "", "text_model", XTest, YTest, "no fusion (text only)", "text_model_score_2")
    evalDecisionModel(mainPath, "", "decision_model", "", "text_model", [XTest, testImgSntmtProbs], YTest, "decision-level fusion", "decision_model_score_2")
    # evalModel(mainPath, "", "cat_ftr-lvl_model", [XTest, testImgCategories], YTest, "image category feature-level fusion", "cat_ftr-lvl_model_score")
    # evalModel(mainPath, "", "cmp_ftr-lvl_model", [XTest, testImgFeatures], YTest, "image components feature-level fusion", "cmp_ftr-lvl_model_score")

if __name__ == "__main__":
    main()
