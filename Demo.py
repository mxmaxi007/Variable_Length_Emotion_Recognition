import sys
import os
import math
import re
import time

import numpy as np
import tensorflow as tf
from PIL import Image

import Preprocess.Spectrogram as Spectrogram
import Preprocess.Load_Data as Load_Data
import Preprocess.Normalization as Normalization

import Model.CNN_Const as CNN_Const
import Model.CNN_LSTM_Const as CNN_LSTM_Const
import Model.CNN_LSTM_Attention_Const as CNN_LSTM_Attention_Const
import Model.CNN_RNN_Var as CNN_RNN_Var
import Model.CNN_RNN_Const as CNN_RNN_Const

import Metrics.Accuracy as Accuracy


def Var_CNN_Output():
    spectrogram_path = "/Users/max/Downloads/Data/Personal/interspeech18/Test_Data/Ses01M_impro01_F024_neu.npy";
    model_dir = "/Users/max/Downloads/Data/Personal/interspeech18/CNN_RNN_Var_3/model";

    spectrogram = np.load(spectrogram_path);
    spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1], 1);
    print(spectrogram.shape);

    emo_dict = {0: "Neutral", 1: "Angry", 2: "Happy", 3: "Sad"};
    emo_num = 4;

    tf.reset_default_graph();
    y_predict = [];

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["CNN_RNN_Var"], model_dir);

        h_pool2 = sess.graph.get_tensor_by_name("pool2/MaxPool:0");

        cnn_out = sess.run(h_pool2, feed_dict={
            "inputs:0": spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2])
        });

    spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1]);
    input_min = spectrogram.min();
    input_max = spectrogram.max();
    spectrogram = 256 * (spectrogram - input_min) / (input_max - input_min);

    cnn_out = cnn_out.sum(axis=3).reshape(cnn_out.shape[1], cnn_out.shape[2]);
    output_min = cnn_out.min();
    output_max = cnn_out.max();
    cnn_out = 256 * (cnn_out - output_min) / (output_max - output_min);

    # input_im = Image.fromarray(spectrogram.T);
    # input_im.show();

    output_im = Image.fromarray(cnn_out.T[::-1, ::-1]);
    # output_im.show();
    if output_im.mode != 'RGB':
        output_im = output_im.convert('RGB');

    output_im.save("/Users/max/Downloads/Data/Personal/interspeech18/Test_Data/cnn_out_var.jpg");


def Image_Convert():
    # print(info["probabilities"][0]);
    spectrogram_img_path = "/Users/max/Downloads/Data/Personal/interspeech18/Test_Data/spectrogram.png";
    spectrogram_im = Image.open(spectrogram_img_path);
    spectrogram_im_mat = np.asarray(spectrogram_im.convert("L"));
    spectrogram_im = Image.fromarray(spectrogram_im_mat[:, ::-1]);
    spectrogram_im.save("/Users/max/Downloads/Data/Personal/interspeech18/Test_Data/spectrogram.jpg");


def Const_CNN_Output():
    spectrogram_vec_path = "/Users/max/Downloads/Data/Personal/interspeech18/Test_Data/Ses01M_impro01_F019_ang_const.npy";
    model_dir = "/Users/max/Downloads/Data/Personal/interspeech18/CNN_RNN_Const_3/model";

    spectrogram_vec = np.load(spectrogram_vec_path);
    spectrogram_vec = spectrogram_vec.reshape(spectrogram_vec.shape[0], spectrogram_vec.shape[1],
                                              spectrogram_vec.shape[2], 1);


    tf.reset_default_graph();

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["CNN_RNN_Const"], model_dir);

        h_pool2 = sess.graph.get_tensor_by_name("pool2/MaxPool:0");

        cnn_out = sess.run(h_pool2, feed_dict={
            "inputs:0": spectrogram_vec
        });

    # print(cnn_out.shape);
    cnn_out = cnn_out.sum(axis=3);
    output_min = cnn_out.min();
    output_max= cnn_out.max();
    cnn_out = 256 * (cnn_out - output_min) / (output_max - output_min);

    output_im_1 = Image.fromarray(cnn_out[0].T[::-1, ::-1]);
    # output_im.show();
    if output_im_1.mode != 'RGB':
        output_im_1 = output_im_1.convert('RGB');

    output_im_1.save("/Users/max/Downloads/Data/Personal/interspeech18/Test_Data/cnn_out_const_1.jpg");

    output_im_2 = Image.fromarray(cnn_out[1].T[::-1, ::-1]);
    # output_im.show();
    if output_im_2.mode != 'RGB':
        output_im_2 = output_im_2.convert('RGB');

    output_im_2.save("/Users/max/Downloads/Data/Personal/interspeech18/Test_Data/cnn_out_const_2.jpg");


def main():
    # if len(sys.argv) != 7:
    #     print('Usage: python3 ' + sys.argv[0] + ' wav_dir_path spectrogram_dir_path output_dir test_session classifer_type spectrogram_type\n');
    #     sys.exit(2);

    start = time.time();

    Var_CNN_Output();
    # Image_Convert();
    # Const_CNN_Output();

    end = time.time();
    print("Total Time: {}s".format(end - start));


if __name__ == "__main__":
    main();
