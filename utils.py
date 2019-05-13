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



# ============================================================================================
# Combinding Features
# ============================================================================================
def combineFeats():
	"""
		Combining audio and doc features.
	"""


	# PROCESSING AUDIO
    # ===============================
    hyperparams = {'exp': 20, 'timesteps': 30, 'stride': 1, 'lr': 9.9999999999999995e-07, 'nlayers': 3, 'hsize': 128, 'batchsize': 128, 'epochs': 300, 'momentum': 0.80000000000000004, 'decay': 0.98999999999999999, 'dropout': 0.20000000000000001, 'dropout_rec': 0.20000000000000001, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3, 'wepochs': 25, 'layertype': 'bi-lstm', 'merge_mode': 'mul', 'dirpath': 'data/LSTM_10-audio/', 'exppath': 'data/LSTM_10-audio/20/', 'text': 'data/Step10/alltext.txt', 'balClass': False}
    exppath = hyperparams['exppath']

    # load model
    with open(exppath + "/model.json", "r") as json_file:
        model_json = json_file.read()
    try:
        model = model_from_json(model_json)
    except:
        model = model_from_json(model_json, custom_objects={'myrelu':myrelu})

    lr = hyperparams['lr']
    loss = hyperparams['loss']
    momentum = hyperparams['momentum']
    nlayers = hyperparams['nlayers']
    # text = 'data/Step10/alltext.txt'

    # load best model and evaluate
    filepath_best = exppath + "/weights-best.hdf5"
    model.load_weights(filepath=filepath_best)
    print('--- load weights')

    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('--- compile model')

    # load data
    X_train, Y_train, X_dev, Y_dev, R_train, R_dev = loadAudio()
    print('--- load data')

    # getting activations from final layer
    layer = model.layers[nlayers-1]
    inputs = [K.learning_phase()] + model.inputs
    _layer2 = K.function(inputs, [layer.output])
    acts_train = np.squeeze(_layer2([0] + [X_train]))
    acts_dev = np.squeeze(_layer2([0] + [X_dev]))
    print('--- got activations')

    # PROCESSING DOCS
    # ===============================
    hyperparams = {'exp': 330, 'timesteps': 7, 'stride': 3, 'lr': 0.10000000000000001, 'nlayers': 2, 'hsize': 4, 'batchsize': 64, 'epochs': 300, 'momentum': 0.84999999999999998, 'decay': 1.0, 'dropout': 0.10000000000000001, 'dropout_rec': 0.80000000000000004, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3, 'wepochs': 25, 'layertype': 'bi-lstm', 'merge_mode': 'concat', 'dirpath': 'data/LSTM_10/', 'exppath': 'data/LSTM_10/330/', 'text': 'data/Step10/alltext.txt', 'balClass': False}
    exppath = hyperparams['exppath']

    # load model
    with open(exppath + "/model.json", "r") as json_file:
        model_json = json_file.read()
    try:
        model = model_from_json(model_json)
    except:
        model = model_from_json(model_json, custom_objects={'myrelu':myrelu})

    lr = hyperparams['lr']
    loss = hyperparams['loss']
    momentum = hyperparams['momentum']
    nlayers = hyperparams['nlayers']

    # load best model and evaluate
    filepath_best = exppath + "/weights-best.hdf5"
    model.load_weights(filepath=filepath_best)
    print('--- load weights')

    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('--- compile model')

    # load data
    X_train_doc, Y_train, X_dev_doc, Y_dev, R_train_doc, R_dev_doc = loadDoc()
    print('--- load data')

    # getting activations from final layer
    layer = model.layers[nlayers - 1]
    inputs = [K.learning_phase()] + model.inputs
    _layer2 = K.function(inputs, [layer.output])
    acts_train_doc = np.squeeze(_layer2([0] + [X_train_doc]))
    acts_dev_doc   = np.squeeze(_layer2([0] + [X_dev_doc]))
    print('--- got activations')

    # FUSE EMBEDDINGS
    # ============================
    acts_train_doc_pad = []
    for idx, subj in enumerate(np.unique(S_train)):
        index = np.where(S_train == subj)[0]
        j = 0
        indexpad = np.where(S_train_doc == subj)[0]
        for i,_ in enumerate(index):
            # print(i)
            if i%4 == 0 and i > 0 and j < indexpad.shape[0]-1:
                j = j+1
            acts_train_doc_pad.append(acts_train_doc[indexpad[j],:])

    acts_dev_doc_pad = []
    for idx, subj in enumerate(np.unique(S_dev)):
        index = np.where(S_dev == subj)[0]
        j = 0
        indexpad = np.where(S_dev_doc == subj)[0]
        for i,_ in enumerate(index):
            # print(i)
            if i%4 == 0 and i > 0 and j < indexpad.shape[0]-1:
                j = j+1
            acts_dev_doc_pad.append(acts_dev_doc[indexpad[j],:])

        # CMVN
        # scaler = preprocessing.StandardScaler().fit(np.asarray(acts_train_doc_pad))
        # acts_train_doc_pad = scaler.transform(np.asarray(acts_train_doc_pad))
        # acts_dev_doc_pad = scaler.transform(np.asarray(acts_dev_doc_pad))

        X_train_fuse = np.hstack((np.asarray(acts_train_doc_pad),acts_train))
        X_dev_fuse = np.hstack((np.asarray(acts_dev_doc_pad),acts_dev))

        # optional
        np.save('data/fuse/X_train.npy', X_train_fuse)
        np.save('data/fuse/features/X_dev.npy', X_dev_fuse)
        np.save('data/fuse/features/Y_train.npy', Y_train)
        np.save('data/fuse/features/Y_dev.npy', Y_dev)
        np.save('data/fuse/features/S_train.npy', S_train)
        np.save('data/fuse/features/S_dev.npy', S_dev)
        np.save('data/fuse/features/R_train.npy', R_train)
        np.save('data/fuse/features/R_dev.npy', R_dev)
