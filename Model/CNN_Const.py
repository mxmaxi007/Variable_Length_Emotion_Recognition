# encoding=UTF-8

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping


def model_design(row_num, col_num, channel, class_num):
    model = Sequential();
    # input: row_num * col_num images -> (row_num, col_num) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (12, 16), activation='relu', input_shape=(row_num, col_num, channel)));
    model.add(MaxPooling2D(pool_size=(2, 2)));
    model.add(Dropout(0.25, seed=1));

    model.add(Conv2D(24, (8, 12), activation='relu'));
    model.add(MaxPooling2D(pool_size=(2, 2)));
    model.add(Dropout(0.25, seed=1));

    model.add(Conv2D(32, (5, 7), activation='relu'));
    model.add(MaxPooling2D(pool_size=(2, 2)));
    model.add(Dropout(0.25, seed=1));

    model.add(Conv2D(40, (3, 3), activation='relu'));
    model.add(MaxPooling2D(pool_size=(2, 2)));
    model.add(Dropout(0.25, seed=1));

    model.add(Conv2D(48, (3, 3), activation='relu'));
    model.add(MaxPooling2D(pool_size=(2, 2)));
    model.add(Dropout(0.25, seed=1));

    model.add(Flatten());
    model.add(Dense(64, activation='relu'));
    model.add(Dropout(0.25, seed=1));
    model.add(Dense(class_num, activation='softmax'));

    # opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True);
    opt = RMSprop(lr=0.001, decay=1e-6);
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']);

    return model;


def model_train(x_train, y_train, class_num, weight_dict, model_file):
    print("CNN_Const Training");
    y_train = keras.utils.to_categorical(y_train, num_classes=class_num);

    print(y_train);
    row_num, col_num, channel = x_train[-1].shape;
    print("Train Sample Num: {}".format(len(x_train)));
    print("Spectrogram Size: ({} {} {})".format(row_num, col_num, channel));

    model = model_design(row_num, col_num, channel, class_num);
    early_stopping = EarlyStopping(monitor='loss', patience=2);
    model.fit(x_train, y_train, batch_size=32, epochs=6, class_weight=weight_dict, callbacks=[early_stopping], shuffle=True);
    # model.fit(x_train, y_train, batch_size=32, epochs=6, callbacks=[early_stopping]);
    model.save(model_file);
    plot_model(model, to_file=model_file.split(".")[0] + ".png");

