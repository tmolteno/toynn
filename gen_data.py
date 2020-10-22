#
#   Generate test and training data for the toy nn examples.
#
#   Licensed under GPLv3. Copyright (c) 2020. Tim Molteno, tim@elec.ac.nz
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def gen_data(t, label, omega, phase, offset, noise_amplitude):
    '''
        Generate some training data
    '''
    noise = np.random.normal(offset, noise_amplitude, t.shape[0])

    if label == 0:
        dat = noise + np.sin(omega*t + phase)
    if label == 1:
        dat = noise + signal.square(omega*t + phase)
    if label == 2:
        dat = noise + signal.sawtooth(omega*t + phase)
    return dat

def gen_dataset(n, signal_length, num_types):
    data = []
    labels = []

    data_types = np.random.randint(0,num_types,n)
    omegas = np.random.uniform(0,10, n)
    phases = np.random.uniform(0,2*np.pi, n)
    offsets = np.random.normal(0,1, n)
    noise_amplitudes = np.random.uniform(0.2,2.0, n)

    t = np.linspace(0,5,signal_length)

    for d, o, p, off, na in zip(data_types, omegas, phases, offsets, noise_amplitudes):
        dat = gen_data(t, label=d, omega=o, phase=p, offset=off, noise_amplitude=na)
        data.append(dat)
        labels.append(d)
    return (data, labels, t)

def plot_data(data, labels, t, num_types):

    def get_data(i, label):
        '''
            Get the ith piece of data that has the specified label.
        '''
        return data[np.nonzero(np.array(labels) == label)[0][i]]

    N_FIG=10
    fig, ax = plt.subplots(N_FIG, num_types, sharex='col', sharey='row')
    for i in range(N_FIG):
        for label in range(num_types):
            ax[i, label].plot(t, get_data(i, label))
    plt.show()

