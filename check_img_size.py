import pandas as pd
from PIL import Image

counter = 0
def getImgSize(path):
    global counter
    counter += 1
    print(counter)
    #print(Image.open(str(path)).size)
    return Image.open(str(path)).size

def saveData(df):
    with open("image_sizes.csv", "w") as writeFile:
        df.to_csv(writeFile, index = False)
        writeFile.close()
def getSize(tup):
    return tup[0] * tup[1]

def main():
    #images = "./existing_image.csv"
    images = "./image_sizes.csv"
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(images, header = 0)
    print(df["SIZE"].mean())
    print(df[df["SIZE"] == df["SIZE"].min()])
    print(df[df["SIZE"] == df["SIZE"].max()])
    #row = df.sample(n = 1).reset_index(drop = True)
    # df["DIM"] = df["IMG"].apply(getImgSize)
    # df["SIZE"] = df["DIM"].apply(getSize)
#     df["IMG_DIMS"] = df["IMG"].apply(getImgSize)
    print(df)
#    saveData(df)
if __name__ == "__main__":
    main()
# filename = os.path.join('path', 'to', 'image', 'file')
# img = Image.open(filename)
# print img.size
