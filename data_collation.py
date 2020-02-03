import os
import re
import csv
import math
import linecache as lc
import tweepy as tw # Installed via pip
import pandas as pd
api_key = "QOp19hmqlyZ4DcU388vHtFUsX"
api_secret = "lxIv8SFUqMSRHxgprkzzSaj8OCP1VRtoVlprjVGADe0qCvMul6"
access_token = "843012610383663105-L4272lsoa1tvS6HXp5sSeGAPlPvMDci"
access_token_secret = "QJrruOYNJZZilBNCQjhotKzRVsJGOraKZqPQgDB6UXUpA"
auth = tw.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)
# DATA CLEANING
#
# MAKE TEST TSV's/CSV's FIRST TO PERFORM ON.
#
# - Create new empty csv to store everything
#
# - Text sentiments tsv already sorted in order of tweet id so binary search can be performed to retrieve their sentiments
#
# 1) Form id list for every 100 tweets by reading tweet id (from the second backslash to the first dash) from first column of b-t4sa - use while loop appending to id list until null or limit of 100 is reached
# 2) Use statuses_lookup with to determine if tweets have been deleted
# 3) If deleted, will not be included in returned tweet list so can use all existing tweets from then
# 4) bin search for text sentiments in other text file and append to row in csv (hence don't bother doing anything with deleted text id)
# Repeat until all tweet id's have been iterated through
#
# Different data items for each image bc it may provide a different result when put through the model

# Binary search
def getTextSntmt(lPoint, rPoint, target): # target is int
    while (lPoint <= rPoint):
        mPoint = int(math.floor(lPoint + ((rPoint - lPoint) / 2)))
        line = lc.getline("./entire datasets/t4sa_text_sentiment.tsv", mPoint).rstrip() # Retrieves line at index indicated by mPoint
        id = int(line.split()[0])
        # print(id)
        # print(target)
        # print(lPoint)
        # print(mPoint)
        # print(rPoint)
        # print("\n")
        if (id == target):
            return line
        elif id < target:
            lPoint = mPoint + 1
        else:
            rPoint = mPoint - 1
        #input()
    return -1 # not found

    # with open("t4sa_text_sentiment.tsv") as readFile:
    #     reader = csv.reader(readFile, delimiter = "\t")
    #     next(reader) # Skips header row
    #     while (lPoint <= rPoint):
    #         mPoint = int(math.floor(lPoint + ((rPoint - lPoint) / 2)))
    #         if ()

#### Figure out a way to collate text content (from raw_tweets use bin search?) and image sentiments into 1 row.
### write 3 csvs, 1 text 1 image 1 both
def existenceCheck(api, idList):
    existingIdsTexts = []
    tweets = api.statuses_lookup(idList)
    for tweet in tweets:
        txtRmvUrl = re.sub(r"https?:\/\/.*[\r\n]*", "", tweet.text, flags=re.MULTILINE).rstrip() # mline in case tweet text encompasses multiple lines
        existingIdsTexts.extend([[tweet.id_str, txtRmvUrl]])
    return existingIdsTexts

def main():
    idList = []
    existingIdsTexts = []
    imageData = {}
    with open("./entire datasets/b-t4sa_all.txt", "r") as readFile:
        for line in readFile: # Image sentiments
            id = re.search(r"(?<=/)\w+(?=-)", line).group(0)
            if (id in imageData):
                imageData[id].append(line.rstrip())
            else:
                imageData[id] = [line.rstrip()]
                idList.extend([id])
                if (len(idList) == 100):
                    existingIdsTexts.extend(existenceCheck(api, idList))
                    idList.clear()
                    #print(f"existingIdsTexts length: {len(existingIdsTexts)}")
                print(f"existingIdsTexts length: {len(existingIdsTexts)}") # Get rid after
        if (len(idList) > 0):
            # lookup
            existingIdsTexts.extend(existenceCheck(api, idList))
            print(f"existingIdsTexts length: {len(existingIdsTexts)}")
        readFile.close()
    # bin search in other file for this ting
    count = 0
    with open("existing_all.csv", "w", newline="") as writeAll, open("existing_text.csv", "w", newline="") as writeText, open("existing_image.csv", "w", newline="") as writeImg:
        writerText = csv.writer(writeText, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        writerText.writerow(["TWID", "NEG", "NEU", "POS", "TEXT"])
        writerImg = csv.writer(writeImg, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        writerImg.writerow(["IMG", "I-SNTMT"])
        writerAll = csv.writer(writeAll, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        writerAll.writerow(["TWID", "NEG", "NEU", "POS", "TEXT", "IMG", "I-SNTMT"])
        for tweet in existingIdsTexts:
            id = tweet[0]
            text = tweet[1]
            result = getTextSntmt(1, 1179957, int(id)).split()
            result.extend([text])
            writerText.writerow(result) # Write text data only
            for image in imageData[id]:
                imgDataList = image.split()
                writerImg.writerow(imgDataList) # Write image data only - Write one row per image in dic list?
                #result.extend(imgDataList)
                writerAll.writerow(result + imgDataList) # Write both data - Write one row per image in dic list?
                count += 1
            # print(result)
            # print(imageData[id])
            print(f"Rows written into all csv: {count}")
        writeAll.close()
        writeText.close()
        writeImg.close()
        print("\n")
        print(f"Unique tweets existing: {len(existingIdsTexts)}")
        print(f"Total rows written into existing_all.csv: {count}")
if __name__ == "__main__":
    main()
