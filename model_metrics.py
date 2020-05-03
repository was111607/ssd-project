import matplotlib.pyplot as plt
import pickle
import re
import numpy as np

def loadHistory(fname, historyList):
    with open("./model histories/training_all/" + fname +".pickle", "rb") as readFile:
        history = pickle.load(readFile)
        readFile.close()
    name = re.search(r"^[a-zA-Z0-9]+", fname).group(0).title()
    historyList.append((name, history))
    return historyList

def main():
    historyList = []
    historyList = loadHistory("decision_model_history", historyList)
    historyList = loadHistory("feature_model_history", historyList)
    historyList = loadHistory("text_model_history", historyList)
    #fig, axs = plt.subplots(len(historyList), 2)
    maxX = 0
    f = plt.figure(figsize=(20, 20))
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)

    names = []
    minLoss = 1.0
    maxLoss = 0.0
    # Losses
    for i in range(len(historyList)):
        names.append(str(historyList[i][0]) + " Train")
        names.append(str(historyList[i][0]) + " Val")
        history = historyList[i][1]
        epochs = len(history["loss"])
        if epochs > maxX:
            maxX = epochs
        curMinTrainLoss = min(history["loss"])
        curMinValLoss = min(history["val_loss"])
        if curMinTrainLoss <= curMinValLoss:
            curMinLoss = curMinTrainLoss
        else:
            curMinLoss = curMinValLoss
        if curMinLoss < minLoss:
            minLoss = round(curMinLoss, 1)
        curMaxTrainLoss = max(history["loss"])
        curMaxValLoss = max(history["val_loss"])
        if curMaxTrainLoss >= curMaxValLoss:
            curMaxLoss = curMaxTrainLoss
        else:
            curMaxLoss = curMaxValLoss
        if curMaxLoss > maxLoss:
            maxLoss = round(curMaxLoss, 1)
        ax1.plot(history["loss"])
        ax1.plot(history["val_loss"])
    ax1.set_title("Model Training and Validation Losses")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend(names, loc = "upper left")
    ax1.set_xticks(range(maxX))
    ax1.set_xticklabels(range(1, maxX + 1))
    ax1.set_yticks(np.arange(0, maxLoss + 0.5, 0.5))

    names = []
    # Accuracies
    minAcc = 1.0
    for i in range(len(historyList)):
        names.append(str(historyList[i][0]) + " Train")
        names.append(str(historyList[i][0]) + " Val")
        history = historyList[i][1]
        epochs = len(history["accuracy"])
        if epochs > maxX:
            maxX = epochs
        curMinTrainAcc = min(history["accuracy"])
        curMinValAcc = min(history["val_accuracy"])
        if curMinTrainAcc <= curMinValAcc:
            curMinAcc = curMinTrainAcc
        else:
            curMinAcc = curMinValAcc
        if curMinAcc < minAcc:
            minAcc = round(curMinAcc, 1)
        ax2.plot(history["accuracy"])
        ax2.plot(history["val_accuracy"])
    ax2.set_title("Model Training and Validation Accuracies")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend(names, loc = "upper left")
    ax2.set_xticks(range(maxX))
    ax2.set_xticklabels(range(1, maxX + 1))
    ax2.set_yticks(np.arange(minAcc - 0.1, 1.1, 0.05))
    plt.show()

if __name__ == "__main__":
    main()
