#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.utils import class_weight
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression as IR
import csv
from scipy import signal
from scipy.stats import kurtosis, skew, spearmanr
import pickle
from sklearn import preprocessing
import pprint

from keras.models import Sequential, model_from_json, Model
from keras.layers import LSTM, Dense, Flatten, Dropout, Bidirectional, concatenate, Input
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, \
    EarlyStopping, ReduceLROnPlateau, Callback
import keras.backend as K


# Let the user know if they are not using a GPU
if K.tensorflow_backend._get_available_gpus() == []:
    print('YOU ARE NOT USING A GPU ON THIS DEVICE! EXITING!')
    # sys.exit()

# ============================================================================================
# Train LSTM Model
# ============================================================================================
def trainLSTM(X_train, Y_train, X_dev, Y_dev, R_train, R_dev, hyperparams):
	"""
		Method to train LSTM model.
		X_{train,dev}: should be [Nexamples, Ntimesteps, Nfeatures]
		Y_{train,dev}: is a vector of binary outcomes
		R_{train,dev}: is the subject ID, useful for later when calculating performance at the subject level
		hyperparams:   is a dict
	"""

    # seed generator
    np.random.seed(1337)

    # grab hyperparamters
    exp         = hyperparams['exp']
    batch_size  = hyperparams['batchsize']
    epochs      = hyperparams['epochs']
    lr          = hyperparams['lr']
    hsize       = hyperparams['hsize']
    nlayers     = hyperparams['nlayers']
    loss        = hyperparams['loss']
    dirpath     = hyperparams['dirpath']
    momentum    = hyperparams['momentum']
    decay       = hyperparams['decay']
    dropout     = hyperparams['dropout']
    dropout_rec = hyperparams['dropout_rec']
    merge_mode  = hyperparams['merge_mode']
    layertype   = hyperparams['layertype']
    balClass    = hyperparams['balClass']
    act_output  = hyperparams['act_output']


    # grab input dimension and number of timesteps (i.e. number of interview sequences)
    dim = X_train.shape[2]
    timesteps = X_train.shape[1]

    # balance training classess
    if balClass:
        cweight = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    else:
        cweight = np.array([1, 1])

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()


    if layertype == 'lstm':
        if nlayers == 1:
            model.add(LSTM(hsize, return_sequences=False, input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 2:
            model.add(LSTM(hsize, return_sequences=True,   input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            model.add(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec))
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 3:
            model.add(LSTM(hsize, return_sequences=True,  input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            model.add(LSTM(hsize, return_sequences=True,  recurrent_dropout=dropout_rec))
            model.add(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec))
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 4:
            model.add(LSTM(hsize, return_sequences=True, input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            model.add(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec,))
            model.add(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec))
            model.add(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec))
            # model.add(Dense(dsize, activation=act_output))

    elif layertype == 'bi-lstm':
        if nlayers == 1:
            model.add(Bidirectional(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec,
                           dropout=dropout), input_shape=(timesteps, dim), merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 2:
            model.add(Bidirectional(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec,
                           dropout=dropout),input_shape=(timesteps, dim), merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec), merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 3:
            model.add(Bidirectional(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec,
                           dropout=dropout),input_shape=(timesteps, dim),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=False,recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 4:
            model.add(Bidirectional(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec,
                           dropout=dropout),input_shape=(timesteps, dim), merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=True,  recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=True,  recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

    # add the final output node
    # this check is useful if training a multiclass model
    if act_output == 'sigmoid':
        dsize = 1
        model.add(Dense(dsize, activation=act_output))

    elif act_output == 'softmax':
        dsize = 27
        model.add(Dense(dsize, activation=act_output))
        Y_train = to_categorical(R_train, num_classes=27)
        Y_dev = to_categorical(R_dev, num_classes=27)

    elif act_output == 'relu':
        dsize = 1
        def myrelu(x):
            return (K.relu(x, alpha=0.0, max_value=27))
        model.add(Dense(dsize, activation=myrelu))
        Y_train = R_train
        Y_dev = R_dev


    # print info on network
    print(model.summary())
    print('--- network has layers:', nlayers, ' hsize:',hsize, ' bsize:', batch_size,
          ' lr:', lr, ' epochs:', epochs, ' loss:', loss, ' act_o:', act_output)

    # define optimizer
    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    # compile model
    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy','mae','mse'])

    # defining callbacks - creating directory to dump files
    dirpath = dirpath + str(exp)
    os.system('mkdir ' + dirpath)

    # serialize model to JSON
    model_json = model.to_json()
    with open(dirpath + "/model.json", "w") as json_file:
        json_file.write(model_json)

    # checkpoints
    # filepaths to checkpoints
    filepath_best       = dirpath + "/weights-best.hdf5"
    filepath_epochs     = dirpath + "/weights-{epoch:02d}-{loss:.2f}.hdf5"

    # log best model
    checkpoint_best     = ModelCheckpoint(filepath_best,   monitor='loss', verbose=0, save_best_only=True, mode='auto')

    # log improved model
    checkpoint_epochs   = ModelCheckpoint(filepath_epochs, monitor='loss', verbose=0, save_best_only=True, mode='auto')
    
    # log results to csv file
    csv_logger          = CSVLogger(dirpath + '/training.log')
    
    # loss_history        = LossHistory()
    # lrate               = LearningRateScheduler()
    
    # update decay as a function of epoch and lr
    lr_decay            = lr_decay_callback(lr, decay)

    # early stopping criterion
    early_stop          = EarlyStopping(monitor='loss', min_delta=1e-04, patience=25, verbose=0, mode='auto')
    # reduce_lr         = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.0001)

    # log files to plot via tensorboard
    tensorboard         = TensorBoard(log_dir=dirpath + '/logs', histogram_freq=0, write_graph=True, write_images=False)
    
    #calculate custom performance metric
    perf                = Metrics()

    # these are the callbacks we care for
    callbacks_list      = [checkpoint_best, checkpoint_epochs, early_stop, lr_decay, tensorboard, csv_logger]

    # train model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_dev, Y_dev),
              class_weight=cweight,
              callbacks=callbacks_list)

    # load best model and evaluate
    model.load_weights(filepath=filepath_best)
    
    # gotta compile it
    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])

    # return predictions of best model
    pred        = model.predict(X_dev,   batch_size=None, verbose=0, steps=None)
    pred_train  = model.predict(X_train, batch_size=None, verbose=0, steps=None)

    return pred, pred_train
