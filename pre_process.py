import os
import re
import tweepy as tw # Installed via pip
import pandas as pd
api_key = "QOp19hmqlyZ4DcU388vHtFUsX"
api_secret = "lxIv8SFUqMSRHxgprkzzSaj8OCP1VRtoVlprjVGADe0qCvMul6 "
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
# 2) Use statuses_lookup with map_=true to determine if tweets have been deleted
# 3) If deleted, status will be empty (only id attribute persists) - do not take any further action
# 4) If not-deleted, bin search for text sentiments in other text file and append to row in csv (hence don't bother doing anything with deleted text id)
# Repeat until all tweet id's have been iterated through
#
# Different data items for each image bc it may provide a different result when put through the model

def main():
    idList = []
    with open("test_bt4sa.txt", "r") as readFile:
        # try:
        for line in readFile:
            id = re.search(r'(?<=/)\w+(?=-)', line)
            #print(id.group(0))
            idList.extend([id.group(0)])
            if (len(idList) == 5): #100
                # for x in range(len(idList)):
                #     print(idList[x])
                idList.clear()
                print("\n")
        if (len(idList) > 0):
            # for x in range(len(idList)):
            #     print(idList[x])
        readFile.close()

    # with open("test_t4sa.tsv", "w") as writeFile:
    #     writer = csv.writer(writeFile, delimiter = "\t")
    #     for line in sortedList:
    #         writer.writerow(line)
    #     writeFile.close()
if __name__ == "__main__":
    main()
