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

def searchId(lPoint, rPoint, target): # target is int
    while (lPoint <= rPoint):
        mPoint = int(math.floor(lPoint + ((rPoint - lPoint) / 2)))
        line = lc.getline("t4sa_text_sentiment.tsv", mPoint).rstrip()
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
    existingIds = []
    tweets = api.statuses_lookup(idList)
    for tweet in tweets:
        existingIds.extend([tweet.id_str])
    return existingIds

def main():
    idList = []
    existingIds = []

    with open("test_bt4sa.txt", "r") as readFile:
        for line in readFile:
            id = re.search(r'(?<=/)\w+(?=-)', line)
            #print(id.group(0))
            idList.extend([id.group(0)])
            if (len(idList) == 100): # change to 100
                existingIds.extend(existenceCheck(api, idList))
                idList.clear()
                print(f"existingIds length: {len(existingIds)}")
        if (len(idList) > 0):
            # lookup
            existingIds.extend(existenceCheck(api, idList))
            print(f"existingIds length: {len(existingIds)}")
        readFile.close()

    # bin search in other file for this ting
    count = 0
    with open("test_write.tsv", "w") as writeFile:
        writer = csv.writer(writeFile, delimiter = "\t")
        for id in existingIds:
            result = searchId(1, 1179957, int(id))
            writer.writerow(result.split()) # 1 skips header
            count += 1
            print(f"number written into tsv: {count}")
        writeFile.close()
if __name__ == "__main__":
    main()
