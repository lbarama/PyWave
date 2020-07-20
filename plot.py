# Author: Jesse Williams
# Company: Global Technology Connection
# Last updated: 2019 12 13

import numpy as np
import matplotlib.pyplot as plt




def plot_channel_histograms(dataset, bins=31, log_plot=True):
    # Organize the channels
    channel0 = []
    channel1 = []
    channel2 = []
    for observation in dataset:
        channel0.append(observation[0])
        channel1.append(observation[1])
        channel2.append(observation[2])
    
    channel0 = np.array(channel0).flatten()
    channel1 = np.array(channel1).flatten()
    channel2 = np.array(channel2).flatten()

    plt.hist(channel0, bins=bins, log=log_plot)
    plt.title('Histogram, channel 1')
    plt.xlabel('Wave amplitude')
    plt.ylabel('Counts')
    plt.show()

    plt.hist(channel1, bins=bins, log=log_plot)
    plt.title('Histogram, channel 2')
    plt.xlabel('Wave amplitude')
    plt.ylabel('Counts')
    plt.show()

    plt.hist(channel2, bins=bins, log=log_plot)
    plt.title('Histogram, channel 3')
    plt.xlabel('Wave amplitude')
    plt.ylabel('Counts')
    plt.show()
