import re
import numpy as np
import pandas as pd
import csv

def saveData(df):
    with open ("train_text_input_subset.csv", "w") as writeFile:
        df.to_csv(writeFile, index = False, quotechar = '"', quoting = csv.QUOTE_ALL)
        writeFile.close()

def main():
    file = "./train_text_input.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(file, header = 0)
    df = df.sample(frac = 0.05).reset_index(drop = True) # Shuffles data
    saveData(df)

if __name__ == "__main__":
    main()
