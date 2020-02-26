import pandas as pd
import numpy as np
from os import path
from keras.models import load_model
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

def main():
    testFile = "./model_input_testing_subset.csv"
    dfTest = pd.read_csv(testFile, header = 0)
    XTest = np.stack(dfTest["TOKENISED"].apply(toArray))
    YTest = dfTest["TXT_SNTMT"].to_numpy("int32")
#    testPaths = dfTest["IMG"].apply(toURL)#.to_numpy("str")
    testImgFeatures = getInputArray(dir + "/image_features_testing40.csv")
    testImgClass = getInputArray(dir + "/image_classifications_testing40.csv")
    fModel = loadModel("feature_model")
    dModel = loadModel("decision_model")
    fModelScore = fModel.evaluate([XTest, testImgFeatures], YTest)
    dModelScore = dModel.evaluate([XTest, testImgClass], YTest)
    print(fModelScore)
    print(dModelScore)

if __name__ == "__main__":
    main()
