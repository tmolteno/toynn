#
#   Minimal Keras example.
#
#   Licensed under GPLv3. Copyright (c) 2020. Tim Molteno, tim@elec.ac.nz
#
import tensorflow as tf
import numpy as np

import sys
sys.path.insert(1, '../')

from gen_data import gen_dataset, plot_data


if __name__=="__main__":
    
    SIGNAL_LENGTH = 512
    TRAINING_N = 150000
    TEST_N = 10000
    N = TEST_N + TRAINING_N
    NUM_TYPES=3

    np.random.seed(101)

    data, labels, t = gen_dataset(N, SIGNAL_LENGTH, NUM_TYPES)

    #plot_data(data, labels, t, NUM_TYPES)

    test_data = np.array(data[0:TEST_N])
    test_labels = np.array(labels[0:TEST_N])

    train_data = np.array(data[TEST_N:N])
    train_labels = np.array(labels[TEST_N:N])

    # reformat labels
    train_labels = tf.keras.utils.to_categorical(train_labels, NUM_TYPES)
    test_labels = tf.keras.utils.to_categorical(test_labels, NUM_TYPES)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(125, activation=tf.nn.relu, input_shape=(SIGNAL_LENGTH,)))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(SIGNAL_LENGTH,)))
    model.add(tf.keras.layers.Dense(NUM_TYPES, activation=tf.nn.softmax))
    model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['categorical_accuracy'])
    model.summary()

    model.fit(train_data, train_labels, epochs=10, batch_size=32)

    loss, test_accuracy = model.evaluate(test_data, test_labels)

    ## Now show a few predictions, and their known true values.
    print("\n\nExample Performance")
    np.set_printoptions(precision=3, suppress=True)
    for i in range(5):
        print("    Truth: {},  Prediction: {}".format(test_labels[i], model.predict(test_data[i:i+1])[0]))
