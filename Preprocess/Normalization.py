# encoding=UTF-8

import sys
import os
import math
import re
import time
import random

import numpy as np


def length_sort(X, Y, sample_weight_list):
    sample_num = len(Y);
    sample_list = [];

    for i in range(sample_num):
        sample_list.append((X[i], Y[i], sample_weight_list[i]));

    random.shuffle(sample_list);
    sample_list = sorted(sample_list, key=lambda x: x[0].shape[0]);

    X = [];
    Y = [];
    sample_weight_list = [];

    for i in range(sample_num):
        X.append(sample_list[i][0]);
        Y.append(sample_list[i][1]);
        sample_weight_list.append(sample_list[i][2]);

    return X, Y, sample_weight_list;


def get_mean_variance(X):
    time_step_num = 0;
    total_num = 0;
    sum_vec = np.zeros([X[0].shape[1], X[0].shape[2]]);

    for spectrogram in X:
        time_step_num += spectrogram.shape[0];
        sum_vec += np.sum(spectrogram, axis=0);
        total_num += spectrogram.shape[0] * spectrogram.shape[1];

    mean_vec = sum_vec / time_step_num;
    mean = sum_vec.sum() / total_num;
    sum_vec = np.zeros([X[0].shape[1], X[0].shape[2]]);

    for spectrogram in X:
        sum_vec += np.sum((spectrogram - mean_vec)**2, axis=0);

    std_vec = np.sqrt(sum_vec / time_step_num);
    std = np.sqrt(sum_vec.sum() / total_num);

    # return mean_vec, std_vec;
    return mean, std;


def normalize(X, mean_vec, std_vec):
    sample_num = len(X);

    for i in range(sample_num):
        X[i] -= mean_vec;
        X[i] /= std_vec;

    return X;


def normalize_list(X_list, mean_vec, std_vec):
    new_X_list = [];

    for X in X_list:
        new_X_list.append(normalize(X, mean_vec, std_vec));

    return new_X_list;