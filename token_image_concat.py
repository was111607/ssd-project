import os
import csv
import pandas as pd
import numpy as np

def saveData(df):
    with open ("existing_model_inputs.csv", "w") as writeFile:
        df.to_csv(writeFile, index=False)
        writeFile.close()

def main():
    file1 = "./existing_text_tokenised.csv"
    file2 = "./existing_image.csv"
    pd.set_option('display.max_colwidth', -1)
    df_tok = pd.read_csv(file1, header = 0,  lineterminator = "\n")
    df_img = pd.read_csv(file2, header = 0,  lineterminator = "\n")
    # df = df.assign(e=pd.Series(np.random.randn(sLength)).values)
    df_tok = df_tok.drop("TEXT", 1)
    df_tok = df_tok.drop("NEW_TEXT", 1)
    df_tok = df_tok.assign(IMG = df_img["IMG"])
    df_tok = df_tok.assign(I_SNTMT = df_img["I-SNTMT\r"])
    saveData(df_tok)

if __name__ == "__main__":
    main()
