
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
# Data Loading
# ============================================================================================
# you will need to point to your data directory
def loadAudio():

	X_train, Y_train = np.load('data/audio/X_train.npy'), np.load('data/audio/Y_train.npy')
	X_dev, Y_dev = np.load('data/audio/X_dev.npy'), np.load('data/audio/Y_dev.npy')
	R_train, R_dev = np.load('data/audio/R_dev.npy'), np.load('data/audio/R_dev.npy')

	return X_train, Y_train, X_dev, Y_dev, R_train, R_dev


def loadDoc():

	X_train, Y_train = np.load('data/doc/X_train.npy'), np.load('data/doc/Y_train.npy')
	X_dev, Y_dev = np.load('data/doc/X_dev.npy'), np.load('data/doc/Y_dev.npy')
	R_train, R_dev = np.load('data/doc/R_dev.npy'), np.load('data/doc/R_dev.npy')

	return X_train, Y_train, X_dev, Y_dev, R_train, R_dev


def loadFuse():

	X_train, Y_train = np.load('data/fuse/X_train.npy'), np.load('data/fuse/Y_train.npy')
	X_dev, Y_dev = np.load('data/fuse/X_dev.npy'), np.load('data/fuse/Y_dev.npy')
	R_train, R_dev = np.load('data/fuse/R_dev.npy'), np.load('data/fuse/R_dev.npy')

	return X_train, Y_train, X_dev, Y_dev, R_train, R_dev
