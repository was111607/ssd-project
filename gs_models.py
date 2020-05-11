"""
--------------------------
Written by William Sewell
--------------------------
Supplies grid search network definitions to network_optimisation.py.

This file does not have to be directly run.
---------------
Files Required
---------------
None

---------------
Files Produced
---------------
None

"""

import pickle
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.merge import concatenate

# Model definitions in order that they were defined/performed,
# The hyperparameters are changed to be input into the next model

def textModel_lstmDropout(dRate = 0.0):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = dRate, recurrent_dropout = 0.5))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def ftrModel_lstmDropout(dRate = 0.0):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
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

def textModel_recDropout(dRate = 0.0):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, recurrent_dropout = dRate))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model

def ftrModel_recDropout(dRate = 0.0):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.4, recurrent_dropout = dRate))(textFtrs)
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

def textModel_x1Dropout(dRate = 0.0):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, recurrent_dropout = 0.4))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(dRate)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def ftrModel_x1Dropout(dRate = 0.0):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.6))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat)
    x1 = Dropout(dRate)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

##############################################################################################################

def textModel_x2Dropout(dRate = 0.0):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, recurrent_dropout = 0.4))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.6)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(dRate)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def ftrModel_x2Dropout(dRate = 0.0):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.6))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat)
    x1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(dRate)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

##############################################################################################################

def textModel_Optimiser(optimiserChoice, lRate):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.5))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    if optimiserChoice == 1:
        optimiser = SGD(lr = lRate, momentum = 0.9)
    else:
        optimiser = Adam(learning_rate = lRate)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def ftrModel_Optimiser(optimiserChoice, lRate):
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
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
        optimiser = SGD(lr = lRate, momentum = 0.9)
    else:
        optimiser = Adam(learning_rate = lRate)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model
