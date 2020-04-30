import pickle
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.merge import concatenate

def textModel_lstmDropout(dRate = 0.0):
    with open("./training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    lstm = Bidirectional(LSTM(embedDim, dropout = dRate, recurrent_dropout = 0.5))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"]) # optimizer = "adam"
    print(model.summary())
    return model

def ftrModel_lstmDropout(dRate = 0.0):
    with open("./training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    lstm = Bidirectional(LSTM(embedDim, dropout = dRate, recurrent_dropout = 0.5))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat)
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

##############################################################################################################

def textModel_Optimiser(optimiserChoice):# (dRate = 0.0): # (lr = 0.0, mom = 0.0): # (dRate = 0.0)
    with open("./training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.5))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    if optimiserChoice == 1:
        optimiser = SGD(lr = 0.0001, momentum = 0.9)
    else:
        optimiser = Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"]) # optimizer = "adam"
    print(model.summary())
    return model

def ftrModel_Optimiser(optimiserChoice): #(lr = 0.0, mom = 0.0): # (dRate): # (extraHLayers)
    with open("./training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.5))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat)
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    if optimiserChoice == 1:
        optimiser = SGD(lr = 0.0001, momentum = 0.9)
    else:
        optimiser = Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model
