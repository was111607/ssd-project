"""
--------------------------
Written by William Sewell
--------------------------
Performs the static data collation step.

This step was executed using the local machine.

---------------
Files Required
---------------
existing_text_train.csv - Stores the training split data for existing_text.csv
                          adding columns storing the filtered and tokenised text.

existing_text_val.csv - Stores the validation split data for existing_text.csv
                        adding columns storing the filtered and tokenised text.

existing_text_test.csv - Stores the testing split data for existing_text.csv
                         adding columns storing the filtered and tokenised text.

existing_all.csv - Stores tweet IDs, neg/neu/pos sentiment polarities, image paths and image sentiments.

b-t4sa_train.txt - Stores the image paths and classified text sentiments for the BT4SA training split.

b-t4sa_val.txt - Stores the image paths and classified text sentiments for the BT4SA validation split.

b-t4sa_test.txt - Stores the image paths and classified text sentiments for the BT4SA testing split.

existing_image_sorted.csv - Stores tweet IDs, image paths and image sentiments in order of ID ascending.

---------------
Files Produced
---------------
model_input_training.csv - Stores training split information used by the models in a single file:
                           The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                           image path, image sentiment, whether the image and text sentiments match.


model_input_training_subset.csv - Stores 50% of the training split information into a single file:
                                  The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                                  image path, image sentiment, whether the image and text sentiments match.

model_input_testing.csv - Stores testing split information used by the models in a single file:
                           The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                           image path, image sentiment, whether the image and text sentiments match.

model_input_validation.csv -Stores validation split information used by the models in a single file:
                          The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                          image path, image sentiment, whether the image and text sentiments match.

"""

import csv
import re
import math
import random
import pandas as pd
import numpy as np
import linecache as lc
from os import path

# Configuration for alternate external directory structure
awsDir = "/media/Data3/sewell"
curDir = "."
isAws = False # Set if on external system
if isAws is True:
    mainPath = awsDir
else:
    mainPath = curDir

# Populate a global dictionary with all tweet IDs as keys that reference lists
# storing the image paths within the passed in split.
def getPaths(fname):
    pathDict = {}
    with open(fname + ".txt", "r") as readFile:
        for line in readFile: # Image paths and sentiments
            path = re.search(r".+(?= )", line).group(0) # Retrieve image path from the first part of each line.
            id = re.search(r"(?<=/)[0-9]+(?=-)", path).group(0) # Retrieve ID from bath between / and -
            if id in pathDict:
                pathDict[id].append(path) # Add to current list if other images have been found and stored for the tweet ID
            else:
                pathDict[id] = [path] # Otherwise intialise a list storing the path and store it into the dictionary
        readFile.close()
    return pathDict

# Create global dictionaries for each split
trainPaths = getPaths(path.join(mainPath, "b-t4sa/b-t4sa_train"))
trainSubsetPaths = getPaths(path.join(mainPath, "b-t4sa/b-t4sa_train"))
valPaths = getPaths(path.join(path.join(mainPath, "b-t4sa/b-t4sa_val")))
testPaths = getPaths(path.join(mainPath, "b-t4sa/b-t4sa_test"))

# Returns a random image path corresponding to the tweet ID.
# Provided the global dictionary name containingthe paths for a given split,
# the list associated to the target ID that stores its related image paths belonging to the
# split is retrieved for which a path is randomly returned.
# Acceptable as both paths will be collected and stored under the same split on conclusion of the step.
def getImgPath(target, dictName):
    pathList = globals()[dictName][target] # Accesses globally stored dictionary given the name
                                           # and returns the stored list value given the target matching the ID key
    randomPath = random.choice(pathList)
    globals()[dictName][target] = pathList.remove(randomPath) # Removes the path from the global dictionary so it
                                                              # cannot be retrieved again
    return randomPath

# Returns the image sentiment by extracting the ID and intra-tweet image index from the inputted target path
# and performs binary search until a line corresponding to the ID is found, which means the same tweet has been found
# but the line may represent a different image.
# Due to the sorting of image paths by ID, the target path is at most 3 lines away from the current line.
# The correct line is retrieved, since the image indexes are in ascending order, by calculating the index offset
# and adding it to the midpoint to retrieve the correct file line number.
# The image sentiment is returned in the end.
def getImgSntmt(lPoint, rPoint, targetPath):
    target = int(re.search(r"(?<=/)\w+(?=-)", targetPath).group(0)) # Extract ID from path
    targetImgNo = int(re.search(r"(?<=-)[0-9]", targetPath).group(0)) # Extract image index from path
    # Perform binary search
    while (lPoint <= rPoint):
        mPoint = int(math.floor(lPoint + ((rPoint - lPoint) / 2)))
        line = lc.getline("./existing_image_sorted.csv", mPoint).rstrip() # Retrieves line at index indicated by mPoint
        lineParts = line.split(",")
        id = int(lineParts[0]) # Extracts ID from path
        # Retrieve image sentiment from the line corresponding to the correct image
        if (id == target):
            path = str(lineParts[1])
            imgNo = int(re.search(r"(?<=-)[0-9]", path).group(0))
            # Correct line number is calculated by adding the offset between the current and target line, represented in index,
            # to the midpoint
            correctLine = lc.getline("./existing_image_sorted.csv", mPoint + (targetImgNo - imgNo)).rstrip()
            lineParts = correctLine.split(",") # Separate data stored in line by comma into a list
            return lineParts[2] # Index of the stored image sentiment
        elif id < target:
            lPoint = mPoint + 1
        else:
            rPoint = mPoint - 1
    raise ValueError # Image sentiment not found

# Classifies the text sentiment by finding the argmax of the inputted polarities.
def getTextSntmt(neg, neu, pos):
    return np.argmax([float(neg), float(neu), float(pos)])

# Returns the result of matching the text and image sentiment.
def matchSntmts(txt, img):
    if int(txt) == int(img):
        return "YES"
    else:
        return "NO"

# Saves the DataFRame to a CSV file with the assigned name.
def saveData(df, fname):
    with open (fname, "w") as writeFile:
        df.to_csv(writeFile, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        writeFile.close()

# Calls individual methods to acquire different static data components:
# Image path, image sentiment, text tentiment and sentiment.
# Before comparing image and text sentiments to check if they match.
def addInputs(df, totalRows, fname, pathDict):
    pd.set_option('display.max_colwidth', -1)
    # Remove unrequired columns from the passed in DataFrame.
    df = df.drop("TEXT", 1)
    df = df.drop("NEW_TEXT", 1)
    # Use each element stored in TWID column into getImgPath to find and store image path into IMG.
    # Lambda resembles the element, which is used as an input into getImgPath to be the search target.
    df["IMG"] = df["TWID"].apply(lambda x: getImgPath(str(x), pathDict))
    # Pass each element stored in IMG column into getImgSntmt individually to find and store image sentiments into IMG_SNTMT.
    df["IMG_SNTMT"] = df["IMG"].apply(lambda x: getImgSntmt(2, totalRows + 1, str(x)))
    df.insert(4, "TXT_SNTMT", None) # Insert a new DataFrame column storing the text sentiment to be calculated.
    df["TXT_SNTMT"] = df.apply(lambda x: getTextSntmt(x["NEG"], x["NEU"], x["POS"]), axis = 1) # Calculates the overall text sentiment.
    df["SNTMT_MATCH"] = df.apply(lambda x: matchSntmts(x["TXT_SNTMT"], x["IMG_SNTMT"]), axis = 1) # Determines if image and text sentiments match.
    saveData(df, fname)

def main():
    # Load split data into individual DataFrames setting header to 1st row of CSV headings and
    # separating on the trailing newline character.
    trainFile = path.join(mainPath, "b-t4sa/existing_text_train.csv")
    valFile = path.join(mainPath, "b-t4sa/existing_text_val.csv")
    testFile = path.join(mainPath, "b-t4sa/existing_text_test.csv")
    trainDf = pd.read_csv(trainFile, header = 0, lineterminator = "\n")
    trainSubDf = trainDf.sample(frac = 0.5).reset_index(drop = True) # Randomly samples 50% of the training split.
    valDf = pd.read_csv(valFile, header = 0, lineterminator = "\n")
    testDf = pd.read_csv(testFile, header = 0, lineterminator = "\n")
    pd.set_option('display.max_colwidth', -1)

    # Establish total row count for binary search of image sentiments.
    allFile = path.join(mainPath, "existing_all.csv")
    totalRows = pd.read_csv(allFile, header = 0, lineterminator = "\n").shape[0]

    # Collecting text and image data together into the provided filenames, also passing
    # in the associated established global path dictionary name to access.
    addInputs(trainDf, totalRows, path.join(mainPath, "b-t4sa/model_input_training.csv"), "trainPaths")
    addInputs(trainSubDf, totalRows, path.join(mainPath, "b-t4sa/model_input_training_subset.csv"), "trainSubsetPaths")
    addInputs(valDf, totalRows, path.join(mainPath, "b-t4sa/model_input_validation.csv"), "valPaths")
    addInputs(testDf, totalRows, path.join(mainPath, "b-t4sa/model_input_testing.csv"), "testPaths")

if __name__ == "__main__":
    main()
