
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


if __name__ == "__main__":


	# 1. load the data for audio
	X_train, Y_train, X_dev, Y_dev, R_train, R_dev = loadAudio()

	# 2. train lstm model
	pred_audio, pred_train_audio = trainLSTM(X_train, Y_train, X_dev, Y_dev, R_train, R_dev, hyperparams)

	# 1b. load the doc data
	X_train, Y_train, X_dev, Y_dev, R_train, R_dev = loadDoc()

	# 2b. train lstm model for doc data
	pred_audio, pred_train_audio = trainLSTM(X_train, Y_train, X_dev, Y_dev, R_train, R_dev, hyperparams)

	# 3. concatenate last layer features for each audio and doc branch.
	combineFeats()
	X_train, Y_train, X_dev, Y_dev, R_train, R_dev = loadFuse()

	# 4. train feedforward.
	# hyperparams can be different (e.g. learning rate, decay, momentum, etc.)
	pred, pred_train = trainHierarchy(X_train_fuse, Y_train, X_dev_fuse, Y_dev, hyperparams)

	# 5. evaluate performance
	f1 = metrics.f1_score(Y_dev, np.round(pred), pos_label=1)
