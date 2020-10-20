#
#   Minimal Keras example.
#
#   Licensed under GPLv3. Copyright (c) 2020. Tim Molteno, tim@elec.ac.nz
#
import tensorflow as tf
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


SIGNAL_LENGTH = 512
TRAINING_N = 150000
TEST_N = 10000
N = TEST_N + TRAINING_N
NUM_TYPES=3

np.random.seed(101)

def gen_data(t, label, omega, phase, offset, noise_amplitude):
    '''
        Generate some training data
    '''
    noise = np.random.normal(offset, noise_amplitude, SIGNAL_LENGTH)

    if label == 0:
        dat = noise + np.sin(omega*t + phase)
    if label == 1:
        dat = noise + signal.square(omega*t + phase)
    if label == 2:
        dat = noise + signal.sawtooth(omega*t + phase)
    return dat

data = []
labels = []

data_types = np.random.randint(0,NUM_TYPES,N)
omegas = np.random.uniform(0,10, N)
phases = np.random.uniform(0,2*np.pi, N)
offsets = np.random.normal(0,1, N)
noise_amplitudes = np.random.uniform(0.2,2.0, N)

t = np.linspace(0,5,SIGNAL_LENGTH)

for d, o, p, off, na in zip(data_types, omegas, phases, offsets, noise_amplitudes):
    dat = gen_data(t, label=d, omega=o, phase=p, offset=off, noise_amplitude=na)
    data.append(dat)
    labels.append(d)
  

# Dataset stuff isn't being used yet

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(64)
dataset = dataset.repeat()


###################################################
def get_data(i, label):
  return data[np.nonzero(data_types == label)[0][i]]

N_FIG=10
fig, ax = plt.subplots(N_FIG, NUM_TYPES, sharex='col', sharey='row')
for i in range(N_FIG):
    for label in range(NUM_TYPES):
        ax[i, label].plot(t, get_data(i, label))
####################################################


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


print(model.predict(test_data[0:2]))
print(test_labels[0:2])
