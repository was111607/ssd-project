"""
--------------------------
Written by William Sewell
--------------------------
Performs the network creation step, supplying network definitions to network_training.py

This file does not have to be run, unless locally to visualise model architectures.
---------------
Files Required
---------------
None

---------------
Files Produced
---------------
OPTIONAL:
Trained model architecture diagramds - Can only be retrieved when using local system.
                                       Visualised and stored using Keras's plot_model method.
"""

import pickle
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Lambda, Bidirectional, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.merge import concatenate
from keras.utils import plot_model
from os import path

# Store model visualisation - can only be ran locally
def visualiseModel(model, fname, overWrite = True):
    print(model.summary())
    if (overWrite is True) or not (path.exists(fname)):
        plot_model(model, to_file=fname)
        print("Visual architecture saved to " + fname)
    else:
        print("Visual architecture already exists for " + fname)

# Self-trained VGG-T4SA model
def sentimentVGG():
    reg = regularizers.l2(0.000005) # / t4sa stated decay / 2
    input = Input(shape = (224, 224, 3))
    x = Conv2D(64, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv1_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(input)
    x = Conv2D(64, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv1_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv2_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(128, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv2_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv3_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(256, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv3_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(256, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv3_3",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(256, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv3_4",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block3_pool")(x)

    # Block 4
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv4_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv4_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv4_3",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv4_4",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv5_1",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv5_2",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv5_3",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = Conv2D(512, (3, 3),
            activation = "relu",
            padding = "same",
            name = "conv5_4",
            bias_regularizer = reg,
            kernel_regularizer = reg,
            trainable = True)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = "block5_pool")(x)
    flatten = Flatten(name = "flatten")(x)
    hidden1 = Dense(4096,
        activation = "relu",
        name = "fc6",
        bias_regularizer = reg,
        kernel_regularizer = reg,
        trainable = True)(flatten)
    dropout1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(4096,
        activation = "relu",
        name = "fc7",
        bias_regularizer = reg,
        kernel_regularizer = reg,
        trainable = True)(dropout1)
    dropout2 = Dropout(0.5)(hidden2)
    output = Dense(3,
        activation = "softmax",
        name = "fc8",
        bias_regularizer = reg,
        kernel_regularizer = reg,
        trainable = True)(dropout2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.001, momentum = 0.9) # learning_rate decays
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

# Template to extend a text-only model to implement decision-level fusion.
# Is not trainable as image classification occurs separately.
def dFusionModel(textModel):
    imageSntmts = Input(shape=(3,), name = "input_2")
    # Calculates the equal-weighted sum of predicted probabilities to be the final score
    output = Lambda(lambda inputs: (inputs[0] / 2) + (inputs[1] / 2))([textModel.output, imageSntmts])
    model = Model(input = [textModel.input, imageSntmts], output = output)
    optimiser = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

###################################

# Arbitrary text model
def textModelArb():
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
    optimiser = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

# Arbitrary feature-level model
def ftrModelArb():
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.4))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat)
    x1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.5)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

###################################
# Text model implementing Adam
def textModelAdam():
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.1, recurrent_dropout = 0.4))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.6)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

# Feature-level model implementing Adam
def ftrModelAdam():
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.4))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat)
    x1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

##########################################
# Grid search-optimised text model
def textModelOpt():
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, recurrent_dropout = 0.4))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.6)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.6)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

# Grid search-optimised feature-level model
def ftrModelOpt():
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1 # ~ 120k
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
    x2 = Dropout(0.4)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

########################################
# Intuitive attempted improvement on a grid search-optimised text model.
# Due to perceived weaknesses of grid search (e.g. 5 epochs on the training subset)
def textModelSelf():
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.1, recurrent_dropout = 0.4))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.6)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

# Intuitive attempted improvement on a grid search-optimised feature-level model.
# Due to perceived weaknesses of grid search (e.g. 5 epochs on the training subset)
def ftrModelSelf():
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.4))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat)
    x1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

########################################

# Intuitive attempted improvement on a grid search-optimised text model with a learning rate of 0.0001
def textModelSelfLr0001():
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.1, recurrent_dropout = 0.4))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm)
    x1 = Dropout(0.6)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

# Intuitive attempted improvement on a grid search-optimised feature-level model with a learning rate of 0.0001
def ftrModelSelfLr0001():
    # Load vocabulary for embedding layer size
    with open("./vocabulary.pickle", "rb") as readFile:
        vocab = pickle.load(readFile)
        maxVocabSize = len(vocab) + 1
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.4))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat)
    x1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1)
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def main():
    return None
    ### Uncomment to visualise, requires pydot and graphviz
    # visualiseModel(dFusionModel(textModel()), "decision_model.png")
    # visualiseModel(model, "image_model_st.png")
    # visualiseModel(ftrModel(), "feature_model.png") ### Uncomment to visualise, requires pydot and graphviz
    # visualiseModel(textModel(), "text_only_model.png")

if __name__ == "__main__":
    main()
