# encoding=UTF-8

import sys
import os
import math
import re
import time
import random

import numpy as np


def judge_label(file_name):
    if "neu" in file_name:
        return 0;
    elif "ang" in file_name:
        return 1;
    elif "hap" in file_name:
        return 2;
    elif "sad" in file_name:
        return 3;
    else:
        return -1;


def load_spectrogram_const(spectrogram_dir_path, session_list, test_session, emo_dict):
    x_train = [];
    y_train = [];
    x_validation = [];
    y_validation = [];
    x_test = [];
    y_test = [];
    train_data = [];
    spectrogram_dir = os.listdir(spectrogram_dir_path);
    sample_num = 0;
    sample_num_vec = np.zeros(len(emo_dict));
    validation_speaker = "";

    for file_name in spectrogram_dir:
        file_path = os.path.join(spectrogram_dir_path, file_name);

        if os.path.isfile(file_path) and re.match(".*.npy", file_name) and ("impro" in file_name):
            spectrogram_vec = np.load(file_path);
            spectrogram_vec = spectrogram_vec.reshape(spectrogram_vec.shape[0], spectrogram_vec.shape[1], spectrogram_vec.shape[2], 1);
            label = judge_label(file_name);
            sample_num += 1;
            sample_num_vec[label] += 1;

            if session_list[test_session] in file_name:
                if validation_speaker == "":
                    validation_speaker = file_name[-12:-8];

                if file_name[-12:-8] == validation_speaker:
                    x_validation.append(spectrogram_vec);
                    y_validation.append(label);
                else:
                    x_test.append(spectrogram_vec);
                    y_test.append(label);
            else:
                for spectrogram in spectrogram_vec:
                    x_train.append(spectrogram);
                    y_train.append(label);

    sample_weight_list = [];
    weight_dict = dict();

    for key in emo_dict:
        weight_dict[key] = sample_num_vec.max() / (sample_num_vec[key] + 0.5);
        print("{} Sample Num: {}".format(emo_dict[key], int(sample_num_vec[key])));

    for label in y_train:
        sample_weight_list.append(weight_dict[label]);

    return x_train, y_train, x_validation, y_validation, x_test, y_test, weight_dict, sample_weight_list;


def load_spectrogram_var(spectrogram_dir_path, session_list, test_session, emo_dict):
    x_train = [];
    y_train = [];
    x_validation = [];
    y_validation = [];
    x_test = [];
    y_test = [];
    spectrogram_dir = os.listdir(spectrogram_dir_path);
    sample_num_vec = np.zeros(len(emo_dict));
    validation_speaker = "";

    for file_name in spectrogram_dir:
        file_path = os.path.join(spectrogram_dir_path, file_name);

        if os.path.isfile(file_path) and re.match(".*.npy", file_name) and ("impro" in file_name):
            spectrogram = np.load(file_path);
            spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1], 1);

            label = judge_label(file_name);
            sample_num_vec[label] += 1;

            if session_list[test_session] in file_name:
                if validation_speaker == "":
                    validation_speaker = file_name[-12:-8];

                if file_name[-12:-8] == validation_speaker:
                    x_validation.append(spectrogram);
                    y_validation.append(label);
                else:
                    x_test.append(spectrogram);
                    y_test.append(label);
            else:
                x_train.append(spectrogram);
                y_train.append(label);

    sample_weight_list = [];
    weight_dict = dict();

    for key in emo_dict:
        weight_dict[key] = sample_num_vec.max() / (sample_num_vec[key] + 0.5);
        print("{} Sample Num: {}".format(emo_dict[key], int(sample_num_vec[key])));

    max_len = max(x_train, key=lambda x: x.shape[0]).shape[0];

    for i in range(len(x_train)):
        sample_weight = weight_dict[y_train[i]] * (max_len / x_train[i].shape[0]);
        sample_weight_list.append(sample_weight);

    # x_train = [x_train[0], x_train[1]];
    # y_train = [y_train[0], y_train[1]];
    # sample_weight_list = [1, 1];

    return x_train, y_train, x_validation, y_validation, x_test, y_test, weight_dict, sample_weight_list;