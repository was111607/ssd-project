"""
--------------------------
Written by William Sewell
--------------------------
Sorts image data by ID as a precursor to the static data collation step.

This step was executed using the local machine.

---------------
Files Required
---------------
existing_image.csv - Stores image paths and image sentiments.

---------------
Files Produced
---------------
existing_image_sorted.csv - Stores tweet IDs, image paths and image sentiments in order of ID ascending.
"""

import re
import pandas as pd

# Saves the sorted DataFrame to a CSV.
def saveData(df):
    with open ("existing_image_sorted.csv", "w") as writeFile:
        df.to_csv(writeFile, index=False)
        writeFile.close()

# Finds and returns ID from passed in image path.
def getID(text):
    path = str(text)
    # Matches on ID that is located between / and -
    return(re.search(r"(?<=/)\w+(?=-)", path).group(0))

def main():
    file = "./existing_image.csv"
    pd.set_option('display.max_colwidth', -1)
    # Create DataFrame of file setting the heading row to the line containing column headings,
    # rows are identified based on trailing newlines
    df = pd.read_csv(file, header = 0, lineterminator = "\n")
    df.insert(0, "ID", None) # Inserts an "ID" column at the first DataFrame column position.
    df["ID"] = df["IMG"].apply(getID) # Populates ID column by extracting ID from the image paths per row.
    df = df.sort_values(by = ["ID"]).reset_index(drop = True) # Sort rows by ID ascending and resets index ordering.
    saveData(df)

if __name__ == "__main__":
    main()
