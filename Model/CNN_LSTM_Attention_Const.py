# encoding=UTF-8

import numpy as np

import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping


def attention_3d_block(inputs, TIME_STEPS):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2]);
    SINGLE_ATTENTION_VECTOR = False;

    a = Permute((2, 1))(inputs);
    # a = Reshape((input_dim, TIME_STEPS))(a);  # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a);

    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a);
        a = RepeatVector(input_dim)(a);

    a_probs = Permute((2, 1), name='attention_vec')(a);
    output_attention_mul = Multiply()([inputs, a_probs]);

    return output_attention_mul;


def model_design(row_num, col_num, channel, class_num):
    # This returns a tensor
    inputs = Input(shape=(row_num, col_num, channel));

    # a layer instance is callable on a tensor, and returns a tensor
    x = Conv2D(16, (12, 16), activation='relu')(inputs);
    x = MaxPooling2D(pool_size=(2, 2))(x);
    x = Dropout(0.25, seed=1)(x);

    x = Conv2D(24, (8, 12), activation='relu')(x);
    x = MaxPooling2D(pool_size=(2, 2))(x);
    x = Dropout(0.25, seed=1)(x);

    x = Conv2D(32, (5, 7), activation='relu')(x);
    x = MaxPooling2D(pool_size=(2, 2))(x);
    x = Dropout(0.25, seed=1)(x);

    x = Permute((2, 1, 3))(x);
    x = Reshape((int(x.shape[1]), int(x.shape[2] * x.shape[3])))(x);

    TIME_STEPS = int(x.shape[1]);

    x = attention_3d_block(x, TIME_STEPS);
    x = LSTM(128)(x);

    # x = LSTM(128, return_sequences=True)(x);
    # x = attention_3d_block(x, TIME_STEPS);
    # x = Flatten()(x);

    x = Dense(64, activation='relu')(x);
    x = Dropout(0.5, seed=1)(x);
    outputs = Dense(class_num, activation='softmax')(x);

    model = Model(inputs=inputs, outputs=outputs);
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True);
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy']);

    return model;


def model_train(x_train, y_train, class_num, weight_dict, model_file):
    print("CNN_LSTM_Attention_Const Training");
    y_train = keras.utils.to_categorical(y_train, num_classes=class_num);

    row_num, col_num, channel = x_train[0].shape;
    print("Train Sample Num: {}".format(len(x_train)));
    print("Spectrogram Size: ({} {} {})".format(row_num, col_num, channel));

    model = model_design(row_num, col_num, channel, class_num);
    early_stopping = EarlyStopping(monitor='loss', patience=2);
    model.fit(x_train, y_train, batch_size=32, epochs=6, class_weight=weight_dict, callbacks=[early_stopping], shuffle=True);
    model.save(model_file);
    plot_model(model, to_file=model_file.split(".")[0] + ".png");
