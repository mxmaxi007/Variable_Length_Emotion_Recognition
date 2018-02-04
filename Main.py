# encoding=UTF-8

import sys
import os
import math
import re
import time

import numpy as np

import Preprocess.Spectrogram as Spectrogram
import Preprocess.Load_Data as Load_Data
import Preprocess.Normalization as Normalization

import Model.CNN_Const as CNN_Const
import Model.CNN_LSTM_Const as CNN_LSTM_Const
import Model.CNN_LSTM_Attention_Const as CNN_LSTM_Attention_Const
import Model.CNN_RNN_Var as CNN_RNN_Var
import Model.CNN_RNN_Const as CNN_RNN_Const

import Metrics.Accuracy as Accuracy


def main():
    if len(sys.argv) != 7:
        print('Usage: python3 ' + sys.argv[0] + ' wav_dir_path spectrogram_dir_path output_dir test_session classifer_type spectrogram_type\n');
        sys.exit(2);

    start = time.time();

    wav_dir_path = sys.argv[1];
    spectrogram_dir_path = sys.argv[2];
    output_dir = sys.argv[3];
    test_session = int(sys.argv[4]);
    classifer_type = sys.argv[5];
    spectrogram_type = sys.argv[6];

    session_list = ["Ses01", "Ses02", "Ses03", "Ses04", "Ses05"];
    emo_dict = {0: "Neutral", 1: "Angry", 2: "Happy", 3: "Sad"};
    emo_num = 4;
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3";

    # Spectrogram.wav_preprocess(wav_dir_path, spectrogram_dir_path, spectrogram_type);
    if spectrogram_type == "Const":
        x_train, y_train, x_validation, y_validation, x_test, y_test, weight_dict, sample_weight_list = Load_Data.load_spectrogram_const(
            spectrogram_dir_path, session_list, test_session, emo_dict);
        x_train, y_train, sample_weight_list = Normalization.length_sort(x_train, y_train, sample_weight_list);
        mean_vec, std_vec = Normalization.get_mean_variance(x_train);

        x_train = Normalization.normalize(x_train, mean_vec, std_vec);
        x_validation = Normalization.normalize_list(x_validation, mean_vec, std_vec);
        x_test = Normalization.normalize_list(x_test, mean_vec, std_vec);

        # model_file = os.path.join(output_dir, classifer_type + "_" + spectrogram_type + ".model");
        #
        # if classifer_type == "CNN":
        #     CNN_Const.model_train(x_train, y_train, emo_num, weight_dict, model_file);
        # elif classifer_type == "CNN_LSTM":
        #     CNN_LSTM_Const.model_train(x_train, y_train, emo_num, weight_dict, model_file);
        # elif classifer_type == "CNN_LSTM_Attention":
        #     CNN_LSTM_Attention_Const.model_train(x_train, y_train, emo_num, weight_dict, model_file);

        re_train = False;

        # if classifer_type == "CNN_RNN":
        #     if re_train:
        #         CNN_RNN_Const.model_re_train(x_train, y_train, x_test, y_test, emo_num, sample_weight_list,
        #                             weight_dict, output_dir);
        #     else:
        #         CNN_RNN_Const.model_train(x_train, y_train, x_test, y_test, emo_num, sample_weight_list,
        #                             weight_dict, output_dir);

        model_dir = os.path.join(output_dir, "model");
        Accuracy.accuracy_const(model_dir, x_test, y_test, emo_num, emo_dict);

    elif spectrogram_type == "Var":
        x_train, y_train, x_validation, y_validation, x_test, y_test, weight_dict, sample_weight_list = Load_Data.load_spectrogram_var(
            spectrogram_dir_path, session_list, test_session, emo_dict);
        x_train, y_train, sample_weight_list = Normalization.length_sort(x_train, y_train, sample_weight_list);
        mean_vec, std_vec = Normalization.get_mean_variance(x_train);

        x_train = Normalization.normalize(x_train, mean_vec, std_vec);
        x_validation = Normalization.normalize(x_validation, mean_vec, std_vec);
        x_test = Normalization.normalize(x_test, mean_vec, std_vec);

        re_train = False;

        if classifer_type == "CNN_RNN":
            if re_train:
                CNN_RNN_Var.model_re_train(x_train, y_train, x_test, y_test, emo_num, sample_weight_list,
                                    weight_dict, output_dir);
            else:
                CNN_RNN_Var.model_train(x_train, y_train, x_test, y_test, emo_num, sample_weight_list,
                                    weight_dict, output_dir);

            # if re_train:
            #     CNN_RNN_Var.model_re_train(x_train, y_train, x_train, y_train, emo_num, sample_weight_list,
            #                                weight_dict, output_dir);
            # else:
            #     CNN_RNN_Var.model_train(x_train, y_train, x_train, y_train, emo_num, sample_weight_list,
            #                             weight_dict, output_dir);

        model_dir = os.path.join(output_dir, "model");
        Accuracy.accuracy_var(model_dir, x_test, y_test, emo_num, emo_dict);
        # Accuracy.accuracy_var(model_dir, x_train, y_train, emo_num, emo_dict);

    end = time.time();
    print("Total Time: {}s".format(end - start));


if __name__ == "__main__":
    main();
