import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from os import path
from keras.models import load_model
from keras.utils import to_categorical
from ast import literal_eval

def toArray(list):
    return np.array(literal_eval(str(list)))

def getInputArray(fname):
    input = pd.read_csv(fname, header = None)
    return input.to_numpy()

def loadModel(fname):
    try:
        model = load_model("./models/" + fname + ".h5")
        return model
    except OSError:
        print("Cannot find model by " + fname + "to load.")

def saveScore(score, fname):
    with open(fname + ".pickle", "wb") as writeFile:
        pickle.dump(score, writeFile)
        writeFile.close()

def main():
    testFile = "./b-t4sa/model_input_testing.csv"
    dfTest = pd.read_csv(testFile, header = 0)
    XTest = np.stack(dfTest["TOKENISED"].apply(toArray))
    YTest = dfTest["TXT_SNTMT"].to_numpy("int32")
#    testPaths = dfTest["IMG"].apply(toURL)#.to_numpy("str")
    testImgFeatures = np.load("./b-t4sa/image features/image_features_testing.npy")
    testImgClass = np.load("./b-t4sa/image classifications/image_classifications_testing.npy")
    #fModel = loadModel("feature_model")
    dModel = loadModel("decision_model")
    tModel = loadModel("text_model")
    #fModelScore = fModel.evaluate([XTest, testImgFeatures], to_categorical(YTest))
    dModelScore = dModel.evaluate([XTest, testImgClass], to_categorical(YTest))
    tModelScore = tModel.evaluate(XTest, to_categorical(YTest))
    #print(f"The loss and accuracy for feature-level fusion is: {fModelScore}")
    print(f"The loss and accuracy for decision-level fusion is: {dModelScore}")
    print(f"The loss and accuracy for no fusion (text only) is: {tModelScore}")
    #saveScore(fModelScore, "feature_model_score")
    saveScore(dModelScore, "decision_model_score")
    saveScore(tModelScore, "text_model_score")

if __name__ == "__main__":
    main()
