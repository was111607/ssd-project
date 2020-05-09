"""
--------------------------
Written by William Sewell
--------------------------
Performs the data cleaning and collation functionalities and was executed using
the local machine.

---------------
Files Required
---------------
t4sa_text_sentiment.tsv - Stores text sentiments and classifed polarities.

b-t4sa_all.txt - Stores image path and classified sentiment for all BT4SA tweets.

---------------
Files Produced
---------------
existing_all.csv - Stores existing tweet IDs, neg/neu/pos sentiment polarities, image paths and image sentiments.

existing_text.csv - Stores existing tweet IDs, neg/neu/pos sentiment polarities.

existing_image.csv - Stores existing image paths and image sentiments.
"""

import re
import csv
import math
import linecache as lc
import tweepy as tw
# Twitter API access data - requires a developer account and app to be created
api_key = "QOp19hmqlyZ4DcU388vHtFUsX"
api_secret = "lxIv8SFUqMSRHxgprkzzSaj8OCP1VRtoVlprjVGADe0qCvMul6"
access_token = "843012610383663105-L4272lsoa1tvS6HXp5sSeGAPlPvMDci"
access_token_secret = "QJrruOYNJZZilBNCQjhotKzRVsJGOraKZqPQgDB6UXUpA"
auth = tw.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Gets text sentiment efficiently by performing binary search and random access
# To match and return file lines storing ID and sentiment polarities (neg/neu/pos)
def getTextSntmt(lPoint, rPoint, target): # target is int
    while (lPoint <= rPoint):
        mPoint = int(math.floor(lPoint + ((rPoint - lPoint) / 2)))
        line = lc.getline("./collation data/t4sa_text_sentiment.tsv", mPoint).rstrip() # Retrieves line at index indicated by mPoint
        id = int(line.split()[0])
        if (id == target):
            return line
        elif id < target:
            lPoint = mPoint + 1
        else:
            rPoint = mPoint - 1
    raise ValueError # Text sentiment not found

# Uses the Tweepy API to check the existence of each tweet associated to the IDS
# passed using idList (in batches of 100) and returns the ID-text component pairs
# for each available returned tweet. The text components have been filtered to
# remove the associated tweet URL.
def existenceCheck(api, idList):
    existingIdsTexts = []
    tweets = api.statuses_lookup(idList)
    for tweet in tweets:
        txtRmvUrl = re.sub(r"https?:\/\/[^\s]*[\r\n]*", "", tweet.text, flags=re.MULTILINE).rstrip() # Checks all lines in tweet to remove tweet URL and trailing characters
        existingIdsTexts.extend([[tweet.id_str, txtRmvUrl]]) # Accumulates existing ID-text pairs to be returned
    return existingIdsTexts

def main():
    idList = [] # Stores up to 100 IDs
    existingIdsTexts = [] # Stores existing ID-text pairs (in list format)
    imageData = {} # Dictionary with key = tweet ID and value = list storing lines from b-t4sa_all.txt (image path <space> sentiment)

    # Builds a dictionary of all tweets and their image data.
    # Also finds text components for all tweet IDs still existing and stored them.
    #
    # Each line in b-t4sa_all.txt is iterated through to retrieve each line
    # containing the image paths and sentiments. If imageData already
    # stores the ID as a key, meaning that a different line for a different
    # image associated to the same tweet is stored, the current line is appended to the list.
    # If not, it resembles the first scraped tweet for the ID and is stored into imageData
    # inside a list.
    # existenceCheck is called when idList accumulates 100 batches for the API
    # and if idList has a leftover batch that is not 100 at the end, it calls
    # existenceCheck. existenceCheck returns the text components associated with
    # the tweet ID in pairs that are accumulated by existingIdsTexts
    with open("./collation data/b-t4sa_all.txt", "r") as readFile:
        for line in readFile:
            id = re.search(r"(?<=/)[0-9]+(?=-)", line).group(0) # Gets tweet ID
            if (id in imageData):
                imageData[id].append(line.rstrip()) # Stores line into current list if data is additional
            else:
                imageData[id] = [line.rstrip()] # Initialises list to be stored if ID has not been processed before
                idList.extend([id])
                if (len(idList) == 100): # Performs existence checks if API batch limit of 100 IDs per request is reached
                    existingIdsTexts.extend(existenceCheck(api, idList))
                    idList.clear()
                print(f"existingIdsTexts length: {len(existingIdsTexts)}") # Number of existing tweets found
        if (len(idList) > 0): # Leftover unfull batch data is existence checked
            # lookup
            existingIdsTexts.extend(existenceCheck(api, idList))
            print(f"existingIdsTexts length: {len(existingIdsTexts)}")
        readFile.close()

    # Builds CSVs storing image and text data components separately and together.
    # Iterates through existing tweets to retrieve ID-associated lines from
    # t4sa_text_sentiment.tsv storing ID and sentiment polarities
    # corresponding to the tweet ID using binary search.
    # Existing image and text data has been collected as a result and is stored into
    # respective CSVs and a unified CSV.
    count = 0
    with open("existing_all.csv", "w", newline="") as writeAll, open("existing_text.csv", "w", newline="") as writeText, open("existing_image.csv", "w", newline="") as writeImg:
        # Initialise files and write columns to organise data components,
        # With all files ONLY storing data for existing tweets.
        # QUOTE_ALL converts all stored data to string that is separated by commas
        writerText = csv.writer(writeText, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        writerText.writerow(["TWID", "NEG", "NEU", "POS", "TEXT"])
        writerImg = csv.writer(writeImg, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        writerImg.writerow(["IMG", "I-SNTMT"])
        writerAll = csv.writer(writeAll, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        writerAll.writerow(["TWID", "NEG", "NEU", "POS", "TEXT", "IMG", "I-SNTMT"])
        # Iterates through list values corresponding to all found existing tweets
        for tweet in existingIdsTexts:
            id = tweet[0]
            text = tweet[1]
            # Returns list storing matching text ID and polarities.
            # Must be provided the start and end indexes of entire data set to
            # correctly perform binary search
            result = getTextSntmt(1, 1179957, int(id)).split()
            result.extend([text]) # Adds text content to the result list
            writerText.writerow(result) # Stores result storing text data only
            # Iterates through all tweet images, stored in separate lines in
            # bt4sa_all.txt, for the current text ID. To record a separate
            # line for each related image
            for image in imageData[id]:
                # Converts image data file line into a list separating each component,
                # i.e Path and classified image sentiment
                imgDataList = image.split()
                writerImg.writerow(imgDataList) # Writes image data only
                writerAll.writerow(result + imgDataList) # Write both data by concatenating the list of their components
                count += 1
            print(f"Rows written into all csv: {count}")
        writeAll.close()
        writeText.close()
        writeImg.close()
        print("\n")
        print(f"Unique tweets existing: {len(existingIdsTexts)}") # Each tweet only has 1 text component
        print(f"Total rows written into existing_all.csv: {count}") # Each tweet can store up to 4 images

if __name__ == "__main__":
    main()
