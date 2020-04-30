import re
import csv
import math
import linecache as lc
import tweepy as tw # Installed via pip
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
        line = lc.getline("./entire datasets/t4sa_text_sentiment.tsv", mPoint).rstrip() # Retrieves line at index indicated by mPoint - lc.getline very fast access time therefore binary search fastest
        id = int(line.split()[0])
        if (id == target):
            return line
        elif id < target:
            lPoint = mPoint + 1
        else:
            rPoint = mPoint - 1
        #input()
    return -1 # not found

#### Figure out a way to collate text content (from raw_tweets use bin search?) and image sentiments into 1 row.
### write 3 csvs, 1 text 1 image 1 both
def existenceCheck(api, idList):
    existingIdsTexts = []
    tweets = api.statuses_lookup(idList)
    for tweet in tweets:
        txtRmvUrl = re.sub(r"https?:\/\/[^\s]*[\r\n]*", "", tweet.text, flags=re.MULTILINE).rstrip() # multiline in case tweet text encompasses multiple lines
        existingIdsTexts.extend([[tweet.id_str, txtRmvUrl]])
    return existingIdsTexts

def main():
    idList = [] # Stores up to 100 IDs
    existingIdsTexts = [] # Stores existing IDs and Texts together (in list format)
    imageData = {} # Dict with key = tweet id and value = list storing strings of lines from b-t4sa all (image path <space> sentiment)

    # Uses Tweepy to retrieve tweet id from B-T4SA images, iterates through all in batches of 100 using Tweepy to check for their existence
    # All existing IDs added to idList - In list format since there may be multiple images for each tweet
    with open("./entire datasets/b-t4sa_all.txt", "r") as readFile:
        for line in readFile: # Image sentiments
            id = re.search(r"(?<=/)[0-9]+(?=-)", line).group(0) #(?<=/)\w+(?=-)
            if (id in imageData):
                imageData[id].append(line.rstrip())
            else:
                imageData[id] = [line.rstrip()]
                idList.extend([id])
                if (len(idList) == 100):
                    existingIdsTexts.extend(existenceCheck(api, idList))
                    idList.clear()
                print(f"existingIdsTexts length: {len(existingIdsTexts)}")
        if (len(idList) > 0):
            # lookup
            existingIdsTexts.extend(existenceCheck(api, idList))
            print(f"existingIdsTexts length: {len(existingIdsTexts)}")
        readFile.close()

    # Builds text, image and combined csvs
    # Line corresponding to tweet id retrieved via binary search (tsv is already sorted) from text sentiments (w/ neg/neu/pos polarities)
    # Write text data directly using this process
    # Write all existing image data (match to id as imageData stores for all)
    # Append both and write into existing_all
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
