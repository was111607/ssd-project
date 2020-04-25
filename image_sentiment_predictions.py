import pandas as pd
import numpy as np
import re
import csv
import os
from os import path
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
from keras.models import load_model

def loadModel(mainPath, fname):
    try:
        modelPath = path.join(mainPath, "models", fname + ".h5")
        model = load_model(modelPath)
        return model
    except OSError:
        print("Cannot find model: " + modelPath + " to load.")
        exit()

def saveResults(dict, mainPath):
    with open(path.join(mainPath, "image_predictions.pickle"), "wb") as writeFile:
        pickle.dump(dict, writeFile)
        writeFile.close()

def saveDataFrame(df, fname):
    with open (fname, "w") as writeFile:
        df.to_csv(writeFile, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        writeFile.close()

def getFilename(path):
    return re.search(r"(?<=/)[0-9]+-[0-9].jpg", path).group(0)

def matchMainModelInput(matchings, df):
    # PErform actions within a dataframe
    df["IMG_PREDS"] = df["IMG"].apply(getFilename).map(matchings)
    return df

def getImgSntmts(mainPath, testLen, modelName, batchSize = 32):
    matchings = {}
    model = loadModel(mainPath, modelName)
    dataGen = ImageDataGenerator(preprocessing_function = preprocess_input)
    print(testGen.filenames)
    dir = path.join(mainPath, "b-t4sa", "data")
    testGen = dataGen.flow_from_directory(path.join(dir, "test"), target_size=(224, 224), batch_size = batchSize, class_mode = None, shuffle = False)
    probs = model.predict_generator(testGen, steps = -(-testLen // batchSize), verbose = 1)
    inputOrder = testGen.filenames
    for imagePath, prob in zip(inputOrder, probs):
        fileName = re.search(r"(?<=/)[0-9]+-[0-9].jpg", imagePath).group(0)
        matchings[fileName] = prob
    saveResults(matchings, mainPath)
    return matchings

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")
    pd.set_option('display.max_colwidth', -1)
    dfTest = pd.read_csv(testFile, header = 0)
    testLen = dfTest.shape[0]
    matchings = getImgSntmts(mainPath, testLen, "img_model")
    updatedDf = matchMainModelInput(matchings, dfTest)
    saveDataFrame(updatedDf, path.join(mainPath, "b-t4sa/model_input_testing_updated.csv"))

if __name__ == "__main__":
    main()