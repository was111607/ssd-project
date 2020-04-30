import shutil
import os
from os import path
import pandas as pd
# For image processing
def initDirs(dir):
    if not path.exists(dir):
        os.makedirs(dir)
        trainDir = path.join(dir, "train")
        os.makedirs(path.join(trainDir, "neg"))
        os.makedirs(path.join(trainDir, "neu"))
        os.makedirs(path.join(trainDir, "pos"))
        trainSubDir = path.join(dir, "train_subset")
        os.makedirs(path.join(trainSubDir, "neg"))
        os.makedirs(path.join(trainSubDir, "neu"))
        os.makedirs(path.join(trainSubDir, "pos"))
        valDir = path.join(dir, "val")
        os.makedirs(path.join(valDir, "neg"))
        os.makedirs(path.join(valDir, "neu"))
        os.makedirs(path.join(valDir, "pos"))
        testDir = path.join(dir, "test")
        os.makedirs(path.join(testDir, "neg"))
        os.makedirs(path.join(testDir, "neu"))
        os.makedirs(path.join(testDir, "pos"))
        print("Initialised image directories")
    print("Data storage already exists")

def saveImgs(paths, newPath, sntmt):
    counter = 0
    newPath = path.join(newPath, sntmt)
    print("Copying images to " + newPath + " for sentiment: " + sntmt)
    for img in paths:
        shutil.copy(img, newPath)
        if (counter % 1000 == 0):
            print(counter)
        counter += 1
    print("Image copying completed for " + newPath + " with sentiment: " + sntmt)

def copyImgs(df, newPath):
    negImgPaths = df.loc[df["TXT_SNTMT"] == 0]["IMG"].tolist()
    saveImgs(negImgPaths, newPath, "neg")
    neuImgPaths = df.loc[df["TXT_SNTMT"] == 1]["IMG"].tolist()
    saveImgs(neuImgPaths, newPath, "neu")
    posImgPaths = df.loc[df["TXT_SNTMT"] == 2]["IMG"].tolist()
    saveImgs(posImgPaths, newPath, "pos")

def produceDirs(dir, file, splitName):
    df = pd.read_csv(file, header = 0)
    pathSntmts = df[["IMG", "TXT_SNTMT"]]
    copyImgs(pathSntmts, path.join(dir, splitName))

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = False
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir
    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv") # append _subset for tuning
    trainSubFile = path.join(mainPath, "b-t4sa/model_input_training_subset.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")
    dir = path.join(mainPath, "b-t4sa", "data")
    initDirs(dir)
    produceDirs(dir, trainFile, "train")
    produceDirs(dir, trainSubFile, "train_subset")
    produceDirs(dir, valFile, "val")
    produceDirs(dir, testFile, "test")


if __name__ == "__main__":
    main()
