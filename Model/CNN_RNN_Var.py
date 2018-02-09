# encoding=UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1);
    return tf.Variable(initial);


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape);
    return tf.Variable(initial);


def conv_mask(inputs):
    used = tf.sign(tf.reduce_max(tf.abs(inputs), axis=[2, 3]));
    length_seq = tf.reduce_sum(used, axis=1);
    length_seq = tf.cast(length_seq, tf.int32);
    length_seq -= 1;
    mask = tf.sequence_mask(length_seq, tf.shape(inputs)[1], dtype=tf.float32);
    mask = tf.expand_dims(tf.expand_dims(mask, 2), 3);
    return mask;


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME');


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME');


def sequence_length(inputs):
    used = tf.sign(tf.reduce_max(tf.abs(inputs), 2));
    length_seq = tf.reduce_sum(used, axis=1);
    length_seq = tf.cast(length_seq, tf.int32);
    return length_seq;


def dynamic_rnn(input_data, hidden_size):
    rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size);
    # rnn_cell = tf.nn.rnn_cell.LSTMCell(hidden_size);

    # # create 2 LSTMCells
    # rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]
    #
    # # create a RNN cell composed sequentially of a number of RNNCells
    # multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    # 'state' is a tensor of shape [batch_size, cell_state_size]
    length_seq = sequence_length(input_data);
    outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, input_data, sequence_length=length_seq, dtype=tf.float32);

    rng = tf.range(0, tf.shape(length_seq)[0]);
    indexes = tf.stack([rng, length_seq - 1], axis=1, name="indexes");
    fw_outputs = tf.gather_nd(outputs[0], indexes);
    bw_outputs = outputs[1][:, 0];

    outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1);

    return outputs_concat;

    # length_seq = sequence_length(input_data);
    # outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data, sequence_length=length_seq, dtype=tf.float32);
    #
    # rng = tf.range(0, tf.shape(length_seq)[0]);
    # indexes = tf.stack([rng, length_seq - 1], axis=1, name="indexes");
    # relevant_outputs = tf.gather_nd(outputs, indexes);
    #
    # return relevant_outputs;


def model_design(class_num, feature_size):
    """Model function for CNN."""

    # Inputs
    inputs = tf.placeholder(tf.float32, shape=[None, None, feature_size, 1], name="inputs");

    with tf.name_scope("conv1"):
        mask_conv1 = conv_mask(inputs);
        W_conv1 = weight_variable([1, 12, 1, 8]);
        b_conv1 = bias_variable([8]);
        h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1) * mask_conv1;

    with tf.name_scope("pool1"):
        h_pool1 = max_pool_2x2(h_conv1);

    with tf.name_scope("conv2"):
        mask_conv2 = conv_mask(h_pool1);
        W_conv2 = weight_variable([1, 8, 8, 12]);
        b_conv2 = bias_variable([12]);
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) * mask_conv2;

    with tf.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2);

    # with tf.name_scope("conv3"):
    #     mask_conv3 = conv_mask(h_pool2);
    #     W_conv3 = weight_variable([1, 5, 8, 16]);
    #     b_conv3 = bias_variable([16]);
    #     h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) * mask_conv3;
    #
    # with tf.name_scope("pool3"):
    #     h_pool3 = max_pool_2x2(h_conv3);

    with tf.name_scope("rnn1"):
        rnn_inputs = h_pool2;
        rnn_inputs_reshape = tf.reshape(rnn_inputs, [tf.shape(rnn_inputs)[0], -1, rnn_inputs.shape[2] * rnn_inputs.shape[3]], name="reshape");
        h_rnn = dynamic_rnn(rnn_inputs_reshape, 128);

    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([int(h_rnn.shape[1]), 64]);
        b_fc1 = bias_variable([64]);
        h_fc1 = tf.nn.relu(tf.matmul(h_rnn, W_fc1) + b_fc1);

    # with tf.name_scope("batch_normalization"):
    #     h_fc1_bn = tf.layers.batch_normalization(h_fc1);

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob");
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);

    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([int(h_fc1_drop.shape[1]), class_num]);
        b_fc2 = bias_variable([class_num]);

        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2;

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, name="classes"),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="probabilities")
    };

    # Outputs
    labels = tf.placeholder(tf.int32, shape=[None], name="labels");

    with tf.name_scope("metrics"):
        # Add evaluation metrics (for EVAL mode)
        eval_acc_ops = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name="accuracy");

    with tf.name_scope("loss"):
        # Sample Weights
        sample_weights = tf.placeholder(tf.float32, shape=[None], name="sample_weights");

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=sample_weights), name="cross_entropy");

    # Configure the Training Op (for TRAIN mode)
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0003);
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);

        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="train_op");

    model = {
        "loss": loss,
        "train_op": train_op
    };

    return model;


def data_set_gen(x_train, y_train, sample_weight_list, time_step_list, x_validation, y_validation, batch_size):
    # Training Data Set Generation
    train_dataset_x = tf.data.Dataset.from_generator(lambda: x_train, tf.float32);
    train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train);
    train_dataset_weight = tf.data.Dataset.from_tensor_slices(sample_weight_list);
    train_dataset_time_step = tf.data.Dataset.from_tensor_slices(time_step_list);
    train_dataset = tf.data.Dataset.zip(
        (train_dataset_x, train_dataset_y, train_dataset_weight, train_dataset_time_step));
    train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=([None, None, None], [], [], []));
    # train_dataset = train_dataset.shuffle(sample_num, seed=1);

    train_iterator = train_dataset.make_initializable_iterator();
    train_next_element = train_iterator.get_next();

    # Validation Data Set Generation
    validation_dataset_x = tf.data.Dataset.from_generator(lambda: x_validation, tf.float32);
    validation_dataset_y = tf.data.Dataset.from_tensor_slices(y_validation);
    validation_dataset = tf.data.Dataset.zip((validation_dataset_x, validation_dataset_y));
    validation_dataset = validation_dataset.padded_batch(batch_size, padded_shapes=([None, None, None], []));

    validation_iterator = validation_dataset.make_initializable_iterator();
    validation_next_element = validation_iterator.get_next();

    return train_iterator, train_next_element, validation_iterator, validation_next_element;


def model_train(x_train, y_train, x_validation, y_validation, class_num, sample_weight_list, weight_dict, output_dir):
    tf.reset_default_graph();
    shutil.rmtree(output_dir, ignore_errors=True);
    os.mkdir(output_dir);

    print("CNN_RNN_Var Training");
    train_sample_num = len(x_train);
    val_sample_num = len(x_validation);
    feature_size = x_train[0].shape[1];

    print("Train Sample Num: {}".format(train_sample_num));
    print("Validation Sample Num: {}".format(val_sample_num));
    print("Feature Size: {}".format(feature_size));
    print("Saving graph to: {}".format(output_dir));

    time_step_list = [];
    for val in x_train:
        time_step_list.append(val.shape[0]);

    class_weights = np.zeros([class_num]);
    for key in weight_dict:
        class_weights[key] = weight_dict[key];

    epoch_num = 100;
    batch_size = 32;

    train_iterator, train_next_element, validation_iterator, validation_next_element = data_set_gen(x_train, y_train,
                                                                                                    sample_weight_list,
                                                                                                    time_step_list,
                                                                                                    x_validation,
                                                                                                    y_validation,
                                                                                                    batch_size);
    model = model_design(class_num, feature_size);

    # saver = tf.train.Saver();
    # ckpt_dir = os.path.join(model_dir, "ckpt");
    log_dir = os.path.join(output_dir, "log");
    model_dir = os.path.join(output_dir, "model");

    train_writer = tf.summary.FileWriter(log_dir);
    train_writer.add_graph(tf.get_default_graph());

    best_val_acc = -1;

    with tf.Session() as sess:
        info = dict();
        sess.run(tf.global_variables_initializer());

        classes = sess.graph.get_tensor_by_name("classes:0");
        probabilities = sess.graph.get_tensor_by_name("probabilities:0");
        prediction = {"classes": classes, "probabilities": probabilities};

        update_op = sess.graph.get_operation_by_name("metrics/accuracy/update_op");
        accuracy = sess.graph.get_tensor_by_name("metrics/accuracy/value:0");

        for i in range(epoch_num):
            val_acc_vec = [];
            sess.run(train_iterator.initializer);
            print("Epoch Num ({}/{})".format(i + 1, epoch_num));

            while True:
                sess.run(tf.local_variables_initializer());
                try:
                    cur_inputs, cur_labels, cur_sample_weights, time_step_vec = sess.run(train_next_element);

                    info = sess.run(model, feed_dict={"inputs:0": cur_inputs, "labels:0": cur_labels,
                                                      "dropout/keep_prob:0": 0.75,
                                                      # "loss/class_weights:0": class_weights,
                                                      "loss/sample_weights:0": cur_sample_weights});

                    sess.run(update_op, feed_dict={"inputs:0": cur_inputs, "labels:0": cur_labels,
                                                  "dropout/keep_prob:0": 1.0})
                    train_acc = sess.run(accuracy, feed_dict={"inputs:0": cur_inputs, "labels:0": cur_labels,
                                                              "dropout/keep_prob:0": 1.0});

                    # print(cur_labels, time_step_vec, cur_inputs.shape);
                    # print(sess.run(probabilities, feed_dict={"inputs:0": cur_inputs,
                    #                                          "dropout/keep_prob:0": 1.0}));

                except tf.errors.OutOfRangeError:
                    break;

                print("\tBatchShape = {}, Loss = {}, Accuracy = {}".format(cur_inputs.shape, info["loss"], train_acc));

                #Validation Accuracy Calculation
                sess.run(tf.local_variables_initializer());
                sess.run(validation_iterator.initializer);
                val_acc = 0;

                while True:
                    try:
                        cur_val_inputs, cur_val_labels = sess.run(validation_next_element);
                        sess.run(update_op, feed_dict={"inputs:0": cur_val_inputs, "labels:0": cur_val_labels,
                                                       "dropout/keep_prob:0": 1.0})
                        val_acc = sess.run(accuracy, feed_dict={"inputs:0": cur_val_inputs, "labels:0": cur_val_labels,
                                                                  "dropout/keep_prob:0": 1.0});

                        # print(sess.run(probabilities, feed_dict={"inputs:0": cur_val_inputs,
                        #                                          "dropout/keep_prob:0": 1.0}));

                    except tf.errors.OutOfRangeError:
                        break;

            # saver.save(sess, os.path.join(ckpt_dir, str(i) + ".ckpt"));

                if val_acc >= best_val_acc:
                    shutil.rmtree(model_dir, ignore_errors=True);
                    builder = tf.saved_model.builder.SavedModelBuilder(model_dir);
                    builder.add_meta_graph_and_variables(sess, tags=["CNN_RNN_Var"]);
                    builder.save();
                    best_val_acc = val_acc;

                # print(y_validation);
                print("\tValidation Accuracy = {}, Best Validation Accuracy = {}".format(val_acc, best_val_acc));
                val_acc_vec.append(val_acc);


def model_re_train(x_train, y_train, x_validation, y_validation, class_num, sample_weight_list, weight_dict, output_dir):
    tf.reset_default_graph();
    print("CNN_RNN_Var Training");
    train_sample_num = len(x_train);
    val_sample_num = len(x_validation);
    feature_size = x_train[0].shape[1];

    print("Train Sample Num: {}".format(train_sample_num));
    print("Validation Sample Num: {}".format(val_sample_num));
    print("Feature Size: {}".format(feature_size));
    print("Saving graph to: {}".format(output_dir));

    time_step_list = [];
    for val in x_train:
        time_step_list.append(val.shape[0]);

    class_weights = np.zeros([class_num]);
    for key in weight_dict:
        class_weights[key] = weight_dict[key];

    epoch_num = 100;
    batch_size = 32;

    train_iterator, train_next_element, validation_iterator, validation_next_element = data_set_gen(x_train, y_train,
                                                                                                    sample_weight_list,
                                                                                                    time_step_list,
                                                                                                    x_validation,
                                                                                                    y_validation,
                                                                                                    batch_size);
    # saver = tf.train.Saver();
    # ckpt_dir = os.path.join(model_dir, "ckpt");
    log_dir = os.path.join(output_dir, "log");
    model_dir = os.path.join(output_dir, "model");

    train_writer = tf.summary.FileWriter(log_dir);
    train_writer.add_graph(tf.get_default_graph());

    best_val_acc = -1;

    with tf.Session() as sess:
        info = dict();
        tf.saved_model.loader.load(sess, ["CNN_RNN_Var"], model_dir);

        classes = sess.graph.get_tensor_by_name("classes:0");
        probabilities = sess.graph.get_tensor_by_name("probabilities:0");
        prediction = {"classes": classes, "probabilities": probabilities};

        loss = sess.graph.get_tensor_by_name("loss/cross_entropy:0");
        train_op = sess.graph.get_operation_by_name("optimizer/train_op");
        model = {"loss": loss, "train_op": train_op};

        update_op = sess.graph.get_operation_by_name("metrics/accuracy/update_op");
        accuracy = sess.graph.get_tensor_by_name("metrics/accuracy/value:0");

        sess.run(tf.local_variables_initializer());
        sess.run(validation_iterator.initializer);

        while True:
            try:
                cur_val_inputs, cur_val_labels = sess.run(validation_next_element);
                sess.run(update_op, feed_dict={"inputs:0": cur_val_inputs, "labels:0": cur_val_labels,
                                               "dropout/keep_prob:0": 1.0});
                best_val_acc = sess.run(accuracy, feed_dict={"inputs:0": cur_val_inputs, "labels:0": cur_val_labels,
                                                        "dropout/keep_prob:0": 1.0});

                # print(sess.run(probabilities, feed_dict={"inputs:0": cur_val_inputs,
                #                                          "dropout/keep_prob:0": 1.0}));

            except tf.errors.OutOfRangeError:
                break;

        for i in range(epoch_num):
            val_acc_vec = [];
            sess.run(train_iterator.initializer);
            print("Epoch Num ({}/{})".format(i + 1, epoch_num));

            while True:
                sess.run(tf.local_variables_initializer());
                try:
                    cur_inputs, cur_labels, cur_sample_weights, time_step_vec = sess.run(train_next_element);
                    # print(cur_labels, time_step_vec, cur_inputs.shape);
                    info = sess.run(model, feed_dict={"inputs:0": cur_inputs, "labels:0": cur_labels,
                                                      "dropout/keep_prob:0": 0.75,
                                                      # "loss/class_weights:0": class_weights,
                                                      "loss/sample_weights:0": cur_sample_weights});

                    sess.run(update_op, feed_dict={"inputs:0": cur_inputs, "labels:0": cur_labels,
                                                   "dropout/keep_prob:0": 1.0});
                    train_acc = sess.run(accuracy, feed_dict={"inputs:0": cur_inputs, "labels:0": cur_labels,
                                                              "dropout/keep_prob:0": 1.0});

                    # print(sess.run(probabilities, feed_dict={"inputs:0": cur_inputs,
                    #                                          "dropout/keep_prob:0": 1.0}));

                except tf.errors.OutOfRangeError:
                    break;

                print("\tBatchShape = {}, Loss = {}, Accuracy = {}".format(cur_inputs.shape, info["loss"], train_acc));

                #Validation Accuracy Calculation
                sess.run(tf.local_variables_initializer());
                sess.run(validation_iterator.initializer);
                val_acc = 0;

                while True:
                    try:
                        cur_val_inputs, cur_val_labels = sess.run(validation_next_element);
                        sess.run(update_op, feed_dict={"inputs:0": cur_val_inputs, "labels:0": cur_val_labels,
                                                       "dropout/keep_prob:0": 1.0})
                        val_acc = sess.run(accuracy, feed_dict={"inputs:0": cur_val_inputs, "labels:0": cur_val_labels,
                                                                  "dropout/keep_prob:0": 1.0});

                        # print(sess.run(probabilities, feed_dict={"inputs:0": cur_val_inputs,
                        #                                          "dropout/keep_prob:0": 1.0}));

                    except tf.errors.OutOfRangeError:
                        break;

                # print(y_validation);
                print("\tValidation Accuracy = {}, Best Validation Accruacy = {}".format(val_acc, best_val_acc));
                val_acc_vec.append(val_acc);

            # saver.save(sess, os.path.join(ckpt_dir, str(i) + ".ckpt"));

                if val_acc >= best_val_acc:
                    shutil.rmtree(model_dir, ignore_errors=True);
                    builder = tf.saved_model.builder.SavedModelBuilder(model_dir);
                    builder.add_meta_graph_and_variables(sess, tags=["CNN_RNN_Var"]);
                    builder.save();
                    best_val_acc = val_acc;

