import csv
import re
import math
import random
import pandas as pd
import numpy as np
import linecache as lc
from os import path

awsDir = "/media/Data3/sewell"
curDir = "."
isAws = False
if isAws is True:
    mainPath = awsDir
else:
    mainPath = curDir

def getPaths(fname):
    pathDict = {}
    with open(fname + ".txt", "r") as readFile:
        for line in readFile: # Image sentiments
            path = re.search(r".+(?= )", line).group(0)
            id = re.search(r"(?<=/)[0-9]+(?=-)", path).group(0)
            if id in pathDict:
                pathDict[id].append(path)
            else:
                pathDict[id] = [path]
        readFile.close()
    #print(idList)
    return pathDict

trainPaths = getPaths(path.join(mainPath, "b-t4sa/b-t4sa_train"))
trainSubsetPaths = getPaths(path.join(mainPath, "b-t4sa/b-t4sa_train"))
valPaths = getPaths(path.join(path.join(mainPath, "b-t4sa/b-t4sa_val")))
testPaths = getPaths(path.join(mainPath, "b-t4sa/b-t4sa_test"))

def imagesBefore(index, target, list):
    #print(lc.getline("./existing_image_sorted.csv", ).rstrip())
    if (index < 2):
        return list
    line = lc.getline("./existing_image_sorted.csv", index).rstrip() # Retrieves line at index indicated by mPoint
    lineParts = line.split(",")
    id = int(lineParts[0])
    if (id != target):
        return list
    else:
        list.append(lineParts)
        return imagesBefore(index - 1, target, list)


def imagesAfter(index, target, list):
    line = lc.getline("./existing_image_sorted.csv", index).rstrip() # Retrieves line at index indicated by mPoint
    if (line == ""):
        return list
    lineParts = line.split(",")
    id = int(lineParts[0])
    if (id != target):
        return list
    else:
        list.append(lineParts)
        return imagesBefore(index + 1, target, list)

def getImgPath(lPoint, rPoint, target, dictName):
    pathList = globals()[dictName][target]
    randomPath = random.choice(pathList)
    globals()[dictName][target] = pathList.remove(randomPath)
    return randomPath

    #raise ValueError # not found

def getImgSntmt(lPoint, rPoint, targetPath):
    target = int(re.search(r"(?<=/)\w+(?=-)", targetPath).group(0))
    targetImgNo = int(re.search(r"(?<=-)[0-9]", targetPath).group(0))
    while (lPoint <= rPoint):
        mPoint = int(math.floor(lPoint + ((rPoint - lPoint) / 2)))
        line = lc.getline("./existing_image_sorted.csv", mPoint).rstrip() # Retrieves line at index indicated by mPoint
        lineParts = line.split(",")
        id = int(lineParts[0])
        if (id == target):
            path = str(lineParts[1])
            imgNo = int(re.search(r"(?<=-)[0-9]", path).group(0))
            # if (str(targetPath == "data/80461/804619315861393408-1.jpg")):
            #     print()
            #     input()
            correctLine = lc.getline("./existing_image_sorted.csv", mPoint + (targetImgNo - imgNo)).rstrip()
            lineParts = correctLine.split(",")
            return lineParts[2]
        elif id < target:
            lPoint = mPoint + 1
        else:
            rPoint = mPoint - 1
    raise ValueError # not found

def getTextSntmt(neg, neu, pos):
    return np.argmax([float(neg), float(neu), float(pos)])

def matchSntmts(txt, img):
    if int(txt) == int(img):
        return "YES"
    else:
        return "NO"

def saveData(df, fname):
    with open (fname, "w") as writeFile:
        df.to_csv(writeFile, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        writeFile.close()

def addInputs(df, totalRows, fname, pathDict):
    pd.set_option('display.max_colwidth', -1)
    #totalRows = allDf.shape[0]
    df = df.drop("TEXT", 1)
    df = df.drop("NEW_TEXT", 1)
    df["IMG"] = df["TWID"].apply(lambda x: getImgPath(2, totalRows + 1, str(x), pathDict))
    df["IMG_SNTMT"] = df["IMG"].apply(lambda x: getImgSntmt(2, totalRows + 1, str(x)))
    df.insert(4, "TXT_SNTMT", None)
    df["TXT_SNTMT"] = df.apply(lambda x: getTextSntmt(x["NEG"], x["NEU"], x["POS"]), axis = 1)
    df["SNTMT_MATCH"] = df.apply(lambda x: matchSntmts(x["TXT_SNTMT"], x["IMG_SNTMT"]), axis = 1)
    saveData(df, fname)

def main():
    trainFile = path.join(mainPath, "b-t4sa/existing_text_train.csv")
    valFile = path.join(mainPath, "b-t4sa/existing_text_val.csv")
    testFile = path.join(mainPath, "b-t4sa/existing_text_test.csv")
    allFile = path.join(mainPath, "existing_all.csv")
    totalRows = pd.read_csv(allFile, header = 0, lineterminator = "\n").shape[0]
    trainDf = pd.read_csv(trainFile, header = 0, lineterminator = "\n")
    trainSubDf = trainDf.sample(frac = 0.5).reset_index(drop = True) # Shuffles data
    valDf = pd.read_csv(valFile, header = 0, lineterminator = "\n")
    testDf = pd.read_csv(testFile, header = 0, lineterminator = "\n")
    pd.set_option('display.max_colwidth', -1)
    addInputs(trainDf, totalRows, path.join(mainPath, "b-t4sa/model_input_training_TEST.csv"), "trainPaths")
    addInputs(trainSubDf, totalRows, path.join(mainPath, "b-t4sa/model_input_training_subset_TEST.csv"), "trainSubsetPaths")
    addInputs(valDf, totalRows, path.join(mainPath, "b-t4sa/model_input_validation_TEST.csv"), "valPaths")
    addInputs(testDf, totalRows, path.join(mainPath, "b-t4sa/model_input_testing_TEST.csv"), "testPaths")

if __name__ == "__main__":
    main()
