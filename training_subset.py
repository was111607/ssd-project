import re
import numpy as np
import pandas as pd
import csv

def saveData(df, fname):
    with open (fname, "w") as writeFile:
        df.to_csv(writeFile, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        writeFile.close()

def main():
    fileTrain = "./model_input_training.csv"
    fileVal =  "./model_input_validation.csv"
    fileTest = "./model_input_testing.csv"
    pd.set_option('display.max_colwidth', -1)
    dfTrain = pd.read_csv(fileTrain, header = 0)
    dfVal = pd.read_csv(fileVal, header = 0)
    dfTest = pd.read_csv(fileTest, header = 0)
    dfTrain = dfTrain.sample(frac = 0.5).reset_index(drop = True) # Shuffles data
    #dfVal = dfVal.sample(frac = 0.4).reset_index(drop = True) # Shuffles data
    #dfTest = dfTest.sample(frac = 0.4).reset_index(drop = True) # Shuffles data
    saveData(dfTrain, "./b-t4sa/model_input_training_subset.csv")
    #saveData(dfVal, "./model_input_validation_subset.csv")
    #saveData(dfTest, "./model_input_testing_subset.csv")

if __name__ == "__main__":
    main()
