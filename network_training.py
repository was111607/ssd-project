import pandas as pd
import pickle
import numpy as np
import os
from os import path
from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Embedding, LSTM, Input, Lambda, Bidirectional, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.applications.vgg19 import preprocess_input
from keras.utils import to_categorical, plot_model
from keras import regularizers
from keras.optimizers import SGD, Adam
from ast import literal_eval
# Load in data as pandas - process images?
# Look into encoding data with one_hot or hashing_trick
# Pad data - find out best pad as it's not 55 - PREPAD, pad as long as longest sequence
# Do text classification and image processing
# When classify together with images, just concatenate vectors together (process images separately)
# Stochastic graident descent?
# split into 70/20/10train test val
# HyperParameter optimisation

# Initially for text
# Objective (optimiser) function, loss function, metrics to define model in Keras
# Investigate activation function
# Embedding layer - LSTM layer - (optional) CNN elements - hidden layer - softmax layer - output layer
# Feature-level fusion: combined image and text vectors within LSTM or before to classify simultaneously
# Decision-level fusion: classify image and text vectors separately, combine within the softmax layer

# Create 3 models separately - image, text embedding and text concat image total model.


# model.add(embedding) - size of vocab (retrieve from pickle file) + 1, output dim - have to tinker around, input_length=55 (size of sequences), set mask_zero to true.
# Add dropout layers into a dense layer with softmax (for neg neu pos classifications) as last layer, takes output of LSTM, maybe add dense before into softmax if needed.
# LOOK INTO RELU, HIDDEN LAYERS = DENSE (ADD RELU HERE)

# Images:
# Either convert optimal caffe model to keras (load weights, create model architecture) or just use VGG19 and predict from keras
# Predict image then, within mode, append to word vector after initial embedding, then reimbed with new dimentions from appending
# Then model proceeds as normal

# Load model without top, flatten output then append to each word vector (total 1920 + 20588 dims)

# Skimage to retrieve and resize from tweet links
# Maybe have to run all programs in succession to be able to run?

# initialise using LearningRateScheduler and add as callback to training if required
def lrScheduler(epoch, lr):
    epochStep = 4
    divStep = 10
    if (epoch % epochStep == 0) and (epoch != 0):
        return lr / divStep
    return lr

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

def loadModel(mainPath, fname):
    try:
        modelPath = path.join(mainPath, "models", fname + ".h5")
        model = load_model(modelPath)
        print(fname + " successfully loaded")
        return model
    except OSError:
        print("Cannot find model: " + modelPath + " to load.")
        exit()

# Features accounted for separately
def visualiseModel(model, fname):
    if not path.exists(fname):
        plot_model(model, to_file=fname)

def textModel():# (dRate = 0.0): # (lr = 0.0, mom = 0.0): # (dRate = 0.0)
    with open("./training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    #textFtrs = Dense(embedDim, use_bias = False)(textFtrs)
    #print(textFtrs.output)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.1, recurrent_dropout = 0.4))(textFtrs)
    hidden1 = Dense(512, activation = "relu")(lstm) # Make similar to feature??
    x1 = Dropout(0.6)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1) # Make similar to feature??
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(input = input, output = output)
    #optimiser = SGD(lr = 0.0001, momentum = 0.9)
    optimiser = Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"]) # optimizer = "adam"
#    visualiseModel(model, "text_only_model.png") ### Uncomment to visualise, requires pydot and graphviz
#    print(model.summary())
    return model

def dFusionModel(mainPath, textModel):# (dRate = 0.0): # (lr = 0.0, mom = 0.0): # (dRate = 0.0)
    imageSntmts = Input(shape=(3,), name = "input_2")
    output = Lambda(lambda inputs: (inputs[0] / 2) + (inputs[1] / 2))([textModel.output, imageSntmts])
    model = Model(input = [textModel.input, imageSntmts], output = output)
    optimiser = SGD(lr = 0.0001, momentum = 0.9)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"]) # optimizer = "adam"
#    visualiseModel(model, "text_only_model.png") ### Uncomment to visualise, requires pydot and graphviz
#    print(model.summary())
#    saveModel(model, mainPath, saveName, overWrite = False) - No need to save as no weights exist in the extra layers
    return model

def ftrModel(): #(lr = 0.0, mom = 0.0): # (dRate): # (extraHLayers)
    with open("./training_counter.pickle", "rb") as readFile:
        tokeniser = pickle.load(readFile)
        maxVocabSize = len(tokeniser) + 1 # ~ 120k
        readFile.close()
    seqLength = 30
    embedDim = 512
    input = Input(shape=(seqLength,))
    textFtrs = Embedding(maxVocabSize, embedDim, input_length = seqLength, mask_zero = True)(input) # Output is 30*512 matrix (each word represented in 64 dimensions) = 1920
    #textFtrs = Dense(embedDim, use_bias = False)(textFtrs)
    #print(textFtrs.output)
    lstm = Bidirectional(LSTM(embedDim, dropout = 0.5, recurrent_dropout = 0.4))(textFtrs)
    imageFtrs = Input(shape=(embedDim,))
    concat = concatenate([lstm, imageFtrs])
    hidden1 = Dense(512, activation = "relu")(concat) # Make similar to feature??
    x1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(256, activation = "relu")(x1) # Make similar to feature??
    x2 = Dropout(0.3)(hidden2)
    output = Dense(3, activation = "softmax")(x2)
    model = Model(inputs = [input, imageFtrs], output = output)
    #optimiser = SGD(lr = 0.001, momentum = 0.9) #(lr = 0.075, momentum = 0.6)
    optimiser = Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimiser, loss = "categorical_crossentropy", metrics = ["accuracy"])
#    visualiseModel(model, "decision_model.png") ### Uncomment to visualise, requires pydot and graphviz
    # print(model.summary())
    return model
#
# def saveData(list, fname):
#     with open(fname, "w") as writeFile:
#         writer = csv.writer(writeFile, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
#         for i in list:
#             writer.writerow(i)
#         writeFile.close()
#     print(fname + " saved")

def saveHistory(fname, history, mainPath):
    dir = path.join(mainPath, "model histories")
    if not path.exists(dir):
        os.makedirs(dir)
    with open(path.join(dir, fname + ".pickle"), "wb") as writeFile:
        pickle.dump(history.history, writeFile)
        writeFile.close()
    print("Saved history for " + fname)

def saveModel(model, mainPath, fname, overWrite = False):
    dir = path.join(mainPath, "models")
    if not path.exists(dir):
        os.makedirs(dir)
    filePath = path.join(dir, fname + ".h5")
    if path.exists(filePath):
        if overWrite is True:
            msg = "Saved, replacing existing file of same name"
            model.save(filePath)
        else:
            msg = "Not saved, model already exists"
    else:
        msg = "Saved"
        model.save(filePath)
    print(fname + " - " + msg)

def toArray(list):
    return np.array(literal_eval(str(list)))

def summariseResults(results):
    means = results.cv_results_["mean_test_score"]
    stds = results.cv_results_["std_test_score"]
    parameters = results.cv_results_["params"]
    print("Best score of %f with parameters %r" % (results.best_score_, results.best_params_))
    for mean, std, parameter in zip(means, stds, parameters):
        print("Score of %f with std of %f with parameters %r" % (mean, std, parameter))

def getCallbacks(scheduleLr, logDir, logName):
    earlyStoppage = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2, verbose = 1)
    logger = CSVLogger(path.join(logDir, logName + ".csv"), append = False, separator = ",")
    callbacks = [earlyStoppage, logger]
    if scheduleLr is True:
        callbacks.append(LearningRateScheduler(lrScheduler, verbose = 1))
    return callbacks

# Train a text only or decision model
def trainMainModel(model, modelName, historyName, logDir, logName, trainInput, YTrain, valInput, YVal, mainPath, scheduleLr = True, batchSize = 16, epochs = 15):
    callbacks = getCallbacks(scheduleLr, logDir, logName)
    modelHistory = model.fit(trainInput, to_categorical(YTrain), validation_data = (valInput, to_categorical(YVal)), epochs = epochs, batch_size = batchSize, callbacks = callbacks)
    saveHistory(historyName, modelHistory, mainPath)
    saveModel(model, mainPath, modelName, overWrite = True)

# Train image model to improve from bt4sa fine tune
def trainImgModel(model, modelName, historyName, logDir, logName, trainLen, valLen, mainPath, scheduleLr = True, batchSize = 16, epochs = 15):
    callbacks = getCallbacks(scheduleLr, logDir, logName)
    dataGen = ImageDataGenerator(preprocessing_function = preprocess_input)
    dir = path.join(mainPath, "b-t4sa", "data")
    trainGen = dataGen.flow_from_directory(path.join(dir, "train"), target_size=(224, 224), batch_size = batchSize)
    valGen = dataGen.flow_from_directory(path.join(dir, "val"), target_size=(224, 224), batch_size = batchSize)
    modelHistory = model.fit_generator(trainGen,
        steps_per_epoch = -(-trainLen // batchSize),
        validation_data = valGen,
        validation_steps = -(-valLen // batchSize),
        epochs = epochs,
        callbacks = callbacks)
    saveHistory(historyName, modelHistory, mainPath)
    saveModel(model, mainPath, modelName, overWrite = True)

def main():
    awsDir = "/media/Data3/sewell"
    curDir = "."
    isAws = True
    if isAws is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set according to CPU to use
        mainPath = awsDir
    else:
        mainPath = curDir

    trainFile = path.join(mainPath, "b-t4sa/model_input_training.csv")
    trainSubFile = path.join(mainPath, "b-t4sa/model_input_training_subset.csv")
    valFile = path.join(mainPath, "b-t4sa/model_input_validation.csv")
    pd.set_option('display.max_colwidth', -1)
    dfTrain = pd.read_csv(trainFile, header = 0)
    dfTrainSub = pd.read_csv(trainSubFile, header = 0)
    dfVal = pd.read_csv(valFile, header = 0)

    XTrain = np.stack(dfTrain["TOKENISED"].apply(toArray)) # CONVERT THIS TO NUMPY ARRAY OF LISTS
    XTrainSub = np.stack(dfTrainSub["TOKENISED"].apply(toArray))
    XVal = np.stack(dfVal["TOKENISED"].apply(toArray))
    YTrain = dfTrain["TXT_SNTMT"].to_numpy("int32")
    YTrainSub = dfTrainSub["TXT_SNTMT"].to_numpy("int32")
    YVal = dfVal["TXT_SNTMT"].to_numpy("int32")

    # Validation on loading from csv or npy directly.
    dir = path.join(mainPath, "b-t4sa", "image sentiment features")
    if not path.exists(dir):
        print("No image data found, please run image_processing.py")
        exit()
    # featureVGG = initFtrVGG(mainPath, "img_model_st")
    # predictAndSave(trainSubPaths, featureVGG, 15, path.join(dir, "image_sntmt_features_training_50"), mainPath, "backup_data")

    trainImgFeatures = np.load(path.join(dir, "image_sntmt_features_training.npy")) # getInputArray # 50 FOR TUNING
    valImgFeatures = np.load(path.join(dir, "image_sntmt_features_validation.npy"))

    logDir = "./logs"
    if not path.exists(logDir):
        os.makedirs(logDir)

    # trainImgModel(sentimentVGG(),
    #     "img_model_st",
    #     "img_model_st_history",
    #     logDir,
    #     "sntmt_ftr-lvl_adam_log",
    #     dfTrain.shape[0],
    #     dfVal.shape[0],
    #     mainPath)

    trainMainModel(textModel(),
        "text_model_adam",
        "text_model_adam_history",
        logDir,
        "text__adam_log",
        XTrain,
        YTrain,
        XVal,
        YVal,
        mainPath,
        scheduleLr = False)

    # trainMainModel(ftrModel(),
    #     "sntmt_ftr-lvl_model_adam",
    #     "sntmt_ftr-lvl_model_adam_history",
    #     logDir,
    #     "sntmt_ftr-lvl_adam_log",
    #     [XTrain, trainImgFeatures],
    #     YTrain,
    #     [XVal, valImgFeatures],
    #     YVal,
    #     mainPath)

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("batch_sizes", results)

    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("dropouts", results)

    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("dropouts", results)

    # lrs = [0.05]
    # moms = [0.0, 0.2, 0.4, 0.6, 0.8]
    # paramGrid = dict(lr = lrs, mom = moms)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("dropouts_005", results)

    # dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # paramGrid = dict(dRate = dropout)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("lstm_dropouts", results, isAws)

    # dropout = [0.6, 0.7, 0.8, 0.9]# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # paramGrid = dict(dRate = dropout)
    # tModel = KerasClassifier(build_fn = textModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = model_selection.GridSearchCV(estimator = tModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # results = grid.fit(XTrain, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("lstm_rec_dropouts_2h", results, isAws)

    # batchSizes = [16, 32, 64, 128, 256]
    # paramGrid = dict(batch_size = batchSizes)
    # dModel = KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5)
    # grid = slms_search.GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_batch_sizes", results, isAws)

    # hiddenLayers = [0, 1]
    # paramGrid = dict(extraHLayers = hiddenLayers)
    # dModel = KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = slms_search.GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_extra_hidden_layers_opt4", results, isAws)

    # lrs = [0.09]
    # moms = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    # paramGrid = dict(lr = lrs, mom = moms)
    # dModel = KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = slms_search.GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_lr_008", results, isAws)

    # dropout = [0.5, 0.6, 0.7, 0.8, 0.9]
    # paramGrid = dict(dRate = dropout)
    # dModel = KerasClassifier(build_fn = catFtrModel, verbose = 1, epochs = 5, batch_size = 16)
    # grid = slms_search.GridSearchCV(estimator = dModel, param_grid = paramGrid, n_jobs = 1, cv = 3)
    # XCombined = np.array([[XTrain[i], trainImgCategories[i]] for i in range(XTrain.shape[0])])
    # results = grid.fit(XCombined, to_categorical(YTrain))
    # summariseResults(results)
    # saveResults("d_h3_dropout_2h", results, isAws)

if __name__ == "__main__":
    main()
