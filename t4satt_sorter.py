import os
import csv
# Sort each row in text sentiments tsv so binary search can be performed to retrieve their sentiments
# with open("test_t4sa.tsv", "r") as readFile:
#     reader = csv.reader(readFile, delimiter = "\t")
#     next(reader) # Skips header row
#     sortedList = sorted(reader, key=lambda row: int(row[0]))
#     readFile.close()
## VERFIED THAT T4SA_TEXT_SENTIMENT IS ALREADY SORTED, CAN PERFORM BINARY SEARCH
with open("t4sa_text_sentiment.tsv", "r") as readFile:
    reader = csv.reader(readFile, delimiter = "\t") # is this needed?
    next(reader) # Skips header row
    first = 0
    for line in readFile:
        if (first != 0):
            if (int(line.split()[0]) < int(prevLine.split()[0])):
                print("NOT SORTED")
        prevLine = line
        first = 1
    #sortedList = sorted(reader, key=lambda row: int(row[0]))
    readFile.close()
# with open("test_t4sa.tsv", "w") as writeFile:
#     writer = csv.writer(writeFile, delimiter = "\t")
#     for line in sortedList:
#         writer.writerow(line)
#     writeFile.close()
# with open("test_t4sa.tsv", "r") as readFile2:
#     reader = csv.reader(readFile2, delimiter = "\t")
#     next(reader) # Skips header row
#     for line in reader:
#         print(line)
#     readFile2.close()
