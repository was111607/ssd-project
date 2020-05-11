"""
--------------------------
Written by William Sewell
--------------------------
Reorganises the image data to permit image_processing_offline.py to run.
Not required for image_processing_online.py (although not recommended to due to time overhead).

This step was executed using the local machine, with gen_data being copied to external systems using SCP.

---------------
Files Required
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

model_input_validation.csv - Stores validation split information used by the models in a single file:
                             The tweet ID, text sentiment polarities, text sentiment, tokenised text,
                             image path, image sentiment, whether the image and text sentiments match.

data - 470,586 images stored in the BT4SA data set, extracted from b-t4sa_imgs.tar.

---------------
Files Produced
---------------

gen_data - 344,389 images of the existing tweets organised into their sentiment classes
         - under the split that they belong to: training, training subset, validation and testing.

"""

import shutil
import os
from os import path
import pandas as pd

# Initialises split directories and sentiment subdirectories to recieve copie data
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
    else:
        print("Data storage already exists")

# Copies images described by their paths stored in a list as a parameter to a new provided
# directory path made up of the split name and sentiment class folder name
def saveImgs(paths, newPath, sntmt):
    counter = 0
    newPath = path.join(newPath, sntmt)
    print("Copying images to " + newPath + " for sentiment: " + sntmt)
    for img in paths:
        shutil.copy(img, newPath) # Copys image at path img across to path newPath
        if (counter % 1000 == 0): # Tracks copying progress
            print(counter)
        counter += 1
    print("Image copying completed for " + newPath + " with sentiment: " + sntmt)

# Divides a split's DataFrame rows into lists storing image paths grouped by sentiment class.
# df.loc fetches rows where the text sentiment is set the the required label, where the
# image paths are extracted and converted to a list to be input into the image copying function.
def copyImgs(df, newPath):
    negImgPaths = df.loc[df["TXT_SNTMT"] == 0]["IMG"].tolist()
    saveImgs(negImgPaths, newPath, "neg")
    neuImgPaths = df.loc[df["TXT_SNTMT"] == 1]["IMG"].tolist()
    saveImgs(neuImgPaths, newPath, "neu")
    posImgPaths = df.loc[df["TXT_SNTMT"] == 2]["IMG"].tolist()
    saveImgs(posImgPaths, newPath, "pos")

# Intakes each split filename and loads the stored data into a DataFrame,
# extracting the columns storing the image paths and text sentiments only
# since the storage of data also depends on their classified sentiment.
# The shrunk DataFrame is passed to copyImgs with the split type to copy images
# by sentiment class into the correct split directory.
def produceDirs(dir, file, splitName):
    df = pd.read_csv(file, header = 0)
    pathSntmts = df[["IMG", "TXT_SNTMT"]]
    copyImgs(pathSntmts, path.join(dir, splitName))

def main():
    # Configuration for alternate external directory structure
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = False # Set if on external system
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to GPU to use
        mainPath = awsDir
    else:
        mainPath = curDir
    # Define filepaths to retrieve split data
    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv")
    trainSubFile = path.join(mainPath, "b-t4sa/model_input_training_subset.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    testFile = path.join(mainPath, "b-t4sa/model_input_testing.csv")
    dir = path.join(mainPath, "b-t4sa", "gen_data") # Define main directory to copy images to
    # Initialise all directories then proceed with image copying process
    initDirs(dir)
    produceDirs(dir, trainFile, "train")
    produceDirs(dir, trainSubFile, "train_subset")
    produceDirs(dir, valFile, "val")
    produceDirs(dir, testFile, "test")

if __name__ == "__main__":
    main()
