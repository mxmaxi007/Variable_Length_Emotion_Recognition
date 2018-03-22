# encoding=UTF-8

import sys
import os
import math
import re
import time
import shutil

import numpy as np
from spectrum import readwav

import tensorflow as tf


def cal_spectrogram(WINDOW_LEN):
    # [batch_size, signal_length]. Both batch_size and signal_length may be unknown.
    signals = tf.placeholder(tf.float32, shape=[None, None], name="signals");

    # `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
    # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length / 2 + 1
    stfts = tf.contrib.signal.stft(signals, frame_length=WINDOW_LEN, frame_step=int(WINDOW_LEN / 4),
                                   fft_length=1600, window_fn=tf.contrib.signal.hamming_window, pad_end=True);

    # A power spectrogram is the squared magnitude of the complex-valued STFT.
    power_spectrograms = tf.real(stfts * tf.conj(stfts));

    # An energy spectrogram is the magnitude of the complex-valued STFT.
    magnitude_spectrograms = tf.abs(stfts);

    log_offset = 1e-6
    log_magnitude_spectrograms = 10 * tf.log(magnitude_spectrograms + log_offset);

    return log_magnitude_spectrograms;


def get_spectrogram(wav_file, seg_sec):
    tf.reset_default_graph();
    data, sample_rate = readwav(wav_file);
    data = data.astype(np.float32);

    sample_num = len(data);  # the number of sample points in the whole speech sentence
    WINDOW_LEN = int(sample_rate * 0.04);  # window length

    print("Sample Rate = {}, Window Length = {}".format(sample_rate, WINDOW_LEN));

    log_magnitude_spectrograms = cal_spectrogram(WINDOW_LEN);

    if seg_sec < 0: # Calculate the spectrogram from the whole wav sentence
        with tf.Session() as sess:
            spectrogram = sess.run(log_magnitude_spectrograms, feed_dict={"signals:0": data.reshape(1, data.shape[0])})[0];

        time_scale, freq_scale = spectrogram.shape;
        print("Spectrogram Shape = ({}, {})".format(time_scale, freq_scale));
        return spectrogram[:, :int(freq_scale / 2)];
    else: # Calculate the spectrogram from some fixed-length speech segments
        SEG_LEN = sample_rate * seg_sec;  # the number of the sample points in a speech segment
        spectrogram_list = [];

        with tf.Session() as sess:
            test_data = np.ones(SEG_LEN);
            spectrogram = sess.run(log_magnitude_spectrograms, feed_dict={"signals:0": test_data.reshape(1, test_data.shape[0])})[0];
            time_scale, freq_scale = spectrogram.shape;
            print("Spectrogram Shape = ({}, {})".format(time_scale, freq_scale));

            for i in range(int(math.ceil(sample_num / SEG_LEN))):
                start = i * SEG_LEN;
                end = (i + 1) * SEG_LEN;

                if end > sample_num:
                    cur_signal = data[start:];
                else:
                    cur_signal = data[start:end];

                if len(cur_signal) <= WINDOW_LEN * 2:
                    continue;

                spectrogram = sess.run(log_magnitude_spectrograms, feed_dict={"signals:0": cur_signal.reshape(1, cur_signal.shape[0])})[0];
                spectrogram = np.vstack((spectrogram, np.zeros((time_scale - spectrogram.shape[0], freq_scale))));  # zero padding

                spectrogram_list.append(spectrogram[:, :int(freq_scale / 2)]);

        return spectrogram_list;


def wav_preprocess(wav_dir_path, spectrogram_dir_path, spectrogram_type):
    shutil.rmtree(spectrogram_dir_path, ignore_errors=True);
    os.mkdir(spectrogram_dir_path);
    wav_dir = os.listdir(wav_dir_path);

    for file_name in wav_dir:
        file_path = os.path.join(wav_dir_path, file_name);

        if os.path.isfile(file_path) and re.match(".*.wav", file_name):
            if spectrogram_type == "Const":
                spectrogram_list = get_spectrogram(file_path, 3);
                spectrogram_file_name = os.path.splitext(file_name)[0] + ".npy";
                np.save(os.path.join(spectrogram_dir_path, spectrogram_file_name), np.array(spectrogram_list));
            elif spectrogram_type == "Var":
                spectrogram = get_spectrogram(file_path, -1);
                spectrogram_file_name = os.path.splitext(file_name)[0] + ".npy";
                np.save(os.path.join(spectrogram_dir_path, spectrogram_file_name), np.array(spectrogram));


def main():
    if len(sys.argv) != 4:
        print('Usage: python3 ' + sys.argv[0] + ' wav_dir_path spectrogram_dir_path spectrogram_type\n');
        sys.exit(2);

    start = time.time();

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3";
    wav_dir_path = sys.argv[1];
    spectrogram_dir_path = sys.argv[2];
    spectrogram_type = sys.argv[3];
    wav_preprocess(wav_dir_path, spectrogram_dir_path, spectrogram_type);

    end = time.time();
    print("Spectrogram Extract Time: {}s".format(end - start));


if __name__ == "__main__":
    main();