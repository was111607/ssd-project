import shutil
import os
from os import path
import pandas as pd
import re

def removeImgs(paths, newPath, sntmt):
    counter = 0
    newPath = path.join(newPath, sntmt)
    print("Removing images from " + newPath + " for sentiment: " + sntmt)
    for img in paths:
    #    print("removing " + img)
        os.remove(path.join(newPath, img))
        if (counter % 100 == 0):
            print(counter)
        counter += 1
    print(str(counter) + " images removed.")
    print("Image removal completed for " + newPath + " with sentiment: " + sntmt)

def getFname(path):
    return re.search(r"(?<=/)[0-9]+-[0-9].jpg", str(path)).group(0)

def findDirs(df, newPath):
    negImgPaths = df.loc[df["TXT_SNTMT"] == 0]["IMG"].apply(getFname).tolist()
    removeImgs(negImgPaths, newPath, "neg")
    neuImgPaths = df.loc[df["TXT_SNTMT"] == 1]["IMG"].apply(getFname).tolist()
    removeImgs(neuImgPaths, newPath, "neu")
    posImgPaths = df.loc[df["TXT_SNTMT"] == 2]["IMG"].apply(getFname).tolist()
    removeImgs(posImgPaths, newPath, "pos")

def rmvFromDir(dir, fileFrom, fileTo, splitName):
    dfFrom = pd.read_csv(fileFrom, header = 0)
    dfTo = pd.read_csv(fileTo, header = 0)
    imagesToRmv = dfFrom[~(dfTo["IMG"] == dfFrom["IMG"])]
    pathSntmts = imagesToRmv[["IMG", "TXT_SNTMT"]]
    findDirs(pathSntmts, path.join(dir, splitName))

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True
    if isAws is True:
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir
    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")

    trainDupFile = path.join(mainPath, "b-t4sa/model_input_training_dup.csv")
    valDupFile = path.join(mainPath, "b-t4sa/model_input_validation_dup.csv")
    testDupFile = path.join(mainPath, "b-t4sa/model_input_testing_dup.csv")

    dir = path.join(mainPath, "b-t4sa", "data")
    rmvFromDir(dir, trainDupFile, trainFile, "train")
    print("yes")
    rmvFromDir(dir, valDupFile, valFile, "val")
    rmvFromDir(dir, testDupFile, testFile, "test")

if __name__ == "__main__":
    main()
