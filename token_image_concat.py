import os
import csv
import re
import math
import random
import pandas as pd
import numpy as np
import linecache as lc

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

def getImgPath(lPoint, rPoint, target):
    while (lPoint <= rPoint):
        mPoint = int(math.floor(lPoint + ((rPoint - lPoint) / 2)))
        line = lc.getline("./existing_image_sorted.csv", mPoint).rstrip() # Retrieves line at index indicated by mPoint
        lineParts = line.split(",")
        id = int(lineParts[0])
        if (id == target):
            imageList = [lineParts]
            imageList.extend(imagesBefore(mPoint - 1, target, []))
            imageList.extend(imagesAfter(mPoint + 1, target, []))
            return random.choice(imageList)[1]
        elif id < target:
            lPoint = mPoint + 1
        else:
            rPoint = mPoint - 1
    raise ValueError # not found

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
        return "NO" # WORK ON THIS

def saveData(df):
    with open ("train_text_input.csv", "w") as writeFile:
        df.to_csv(writeFile, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        writeFile.close()

def main():
    trainFile = "./existing_text_train.csv"
    allFile = "./existing_all.csv"
    imageFile = "./existing_image_sorted.csv"
    pd.set_option('display.max_colwidth', -1)
    dfTok = pd.read_csv(trainFile, header = 0, lineterminator = "\n")
    dfAll = pd.read_csv(allFile, header = 0, lineterminator = "\n")
    dfImg = pd.read_csv(imageFile, header = 0, lineterminator = "\n")
    totalRows = dfAll.shape[0]
    dfTok = dfTok.drop("TEXT", 1)
    dfTok = dfTok.drop("NEW_TEXT", 1)
    dfTok["IMG"] = dfTok["TWID"].apply(lambda x: getImgPath(2, totalRows + 1, int(x)))
    dfTok["IMG_SNTMT"] = dfTok["IMG"].apply(lambda x: getImgSntmt(2, totalRows + 1, str(x)))
    dfTok.insert(4, "TXT_SNTMT", None)
    dfTok["TXT_SNTMT"] = dfTok.apply(lambda x: getTextSntmt(x["NEG"], x["NEU"], x["POS"]), axis = 1)
    dfTok["SNTMT_MATCH"] = dfTok.apply(lambda x: matchSntmts(x["TXT_SNTMT"], x["IMG_SNTMT"]), axis = 1)
    #print(dfTok)
    saveData(dfTok)

if __name__ == "__main__":
    main()
