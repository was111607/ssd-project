import re
import numpy as np
from os import path
import pickle
import pandas as pd

def getFilename(path):
    return re.search(r"(?<=/)[0-9]+-[0-9].jpg", path).group(0)

def matchPreds(matchings, df):
    return df["IMG"].apply(getFilename).map(matchings)

with open("/media/Data3/sewell/image_predictions_backup.pickle", "rb") as rf:
    a = pickle.load(rf)

trainSubFile = path.join("/media/Data3/sewell/b-t4sa/model_input_training_subset.csv")
pd.set_option('display.max_colwidth', -1)
df = pd.read_csv(trainSubFile, header = 0)
print(df.shape)
# print(mp)
print(mp.values[:10])
# print(np.stack(mp))
# print(np.stack(mp)[:7])
