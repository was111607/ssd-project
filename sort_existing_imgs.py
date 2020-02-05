import os
import csv
import re
import pandas as pd
import numpy as np

def saveData(df):
    with open ("existing_image_sorted.csv", "w") as writeFile:
        df.to_csv(writeFile, index=False)
        writeFile.close()

def getID(text):
    path = str(text)
    return(re.search(r"(?<=/)\w+(?=-)", path).group(0))

def main():
    file = "./existing_image.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header = 0, lineterminator = "\n")
    df.insert(0, "ID", None)
    df["ID"] = df["IMG"].apply(getID)
    df = df.sort_values(by = ["IMG"]).reset_index(drop = True)
    saveData(df)

if __name__ == "__main__":
    main()
