# encoding=UTF-8

import numpy as np

import tensorflow as tf

import keras
from keras.models import load_model
from keras import metrics


def metrics_report(y_predict, y_test, emo_num, emo_dict):
    total_num = len(y_test);
    right_num = 0;
    right_num_vec = np.zeros(emo_num);
    total_num_vec = np.zeros(emo_num);
    confusion_mat = np.zeros((emo_num, emo_num));

    for i in range(total_num):
        if y_predict[i] == y_test[i]:
            right_num += 1;
            right_num_vec[y_test[i]] += 1;

        confusion_mat[y_test[i]][y_predict[i]] += 1;
        total_num_vec[y_test[i]] += 1;

    print("Testing Sample Num: {}".format(total_num));
    print("Predict: {}".format(y_predict));
    print("Label: {}".format(y_test));

    emo_acc_vec = np.zeros(emo_num);

    for key in emo_dict:
        if total_num_vec[key] == 0:
            emo_acc_vec[key] = 0;
        else:
            emo_acc_vec[key] = right_num_vec[key] / total_num_vec[key];

        # print("{} Sample Num: {}".format(emo_dict[key], int(total_num_vec[key])));
        print("{} Accuracy: {}".format(emo_dict[key], emo_acc_vec[key]));

    if total_num == 0:
        weighted_acc = 0;
    else:
        weighted_acc = right_num / total_num;

    unweighted_acc = emo_acc_vec.mean();

    print("Weighted Accuracy: {}".format(weighted_acc));
    print("Unweighted Accuracy: {}".format(unweighted_acc));

    confusion_mat /= total_num_vec.reshape(total_num_vec.shape[0], 1);
    print("Confusion Matrix:\n{}".format(confusion_mat));


def accuracy_const(model_dir, x_test, y_test, emo_num, emo_dict):
    tf.reset_default_graph();
    sample_num = len(x_test);
    y_predict = [];

    with tf.Session() as sess:
        info = dict();
        tf.saved_model.loader.load(sess, ["CNN_RNN_Const"], model_dir);

        classes = sess.graph.get_tensor_by_name("classes:0");
        probabilities = sess.graph.get_tensor_by_name("probabilities:0");
        prediction = {"classes": classes, "probabilities": probabilities};

        for i in range(sample_num):
            info = sess.run(prediction, feed_dict={
                "inputs:0": x_test[i],
                "dropout/keep_prob:0": 1.0});

            prob = info["probabilities"];
            print(prob);
            predict = prob.sum(axis=0).argmax();
            y_predict.append(predict);

    metrics_report(y_predict, y_test, emo_num, emo_dict);


def accuracy_var(model_dir, x_test, y_test, emo_num, emo_dict):
    tf.reset_default_graph();
    sample_num = len(x_test);
    y_predict = [];

    with tf.Session() as sess:
        info = dict();
        tf.saved_model.loader.load(sess, ["CNN_RNN_Var"], model_dir);

        classes = sess.graph.get_tensor_by_name("classes:0");
        probabilities = sess.graph.get_tensor_by_name("probabilities:0");
        prediction = {"classes": classes, "probabilities": probabilities};

        for i in range(sample_num):
            info = sess.run(prediction, feed_dict={
                "inputs:0": x_test[i].reshape(1, x_test[i].shape[0], x_test[i].shape[1], x_test[i].shape[2]),
                "dropout/keep_prob:0": 1.0});

            y_predict.append(info["classes"][0]);
            # print(info["probabilities"][0]);

    metrics_report(y_predict, y_test, emo_num, emo_dict);
