# Author: Jesse Williams
# Company: Global Technology Connection
# Last updated: LBarama 2020 05 20

import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def process_data(dataset,sampling_rate=20,decimals=3, # General inputs
    taper_fraction=0.01, # taper function inputs
    highpass_cutoff=1,highpass_order=4, # high pass filter inputs
    lowpass_cutoff=5, lowpass_order=4): # low pass filter inputs
    """Processes a dataset of shape (n_observations, n_channels, n_readings)
    For each wave it applies:
    centering -> taper -> highpass -> lowpass"""
    
    # By copying the dataset, the orginal dataset will remain unaltered
    dataset = dataset.copy()

    # Process data by each observation and wave
    for idx, observation in enumerate(dataset):
        channel0 = observation[0]
        channel0 = center(channel0)
        channel0 = taper(channel0, taper_fraction)
        channel0 = highpass_filter(channel0, highpass_cutoff, sampling_rate, highpass_order)
        channel0 = lowpass_filter(channel0, lowpass_cutoff, sampling_rate, lowpass_order)
        dataset[idx][0] = channel0
        
        channel1 = observation[1]
        channel1 = center(channel1)
        channel1 = taper(channel1, taper_fraction)
        channel1 = highpass_filter(channel1, highpass_cutoff, sampling_rate, highpass_order)
        channel1 = lowpass_filter(channel1, lowpass_cutoff, sampling_rate, lowpass_order)
        dataset[idx][1] = channel1
        
        channel2 = observation[2]
        channel2 = center(channel2)
        channel2 = taper(channel2, taper_fraction)
        channel2 = highpass_filter(channel2, highpass_cutoff, sampling_rate, highpass_order)
        channel2 = lowpass_filter(channel2, lowpass_cutoff, sampling_rate, lowpass_order)
        dataset[idx][2] = channel2

    # Round the dataset to reduce the size of the array
    #dataset = np.around(dataset,decimals=decimals)
    
    return dataset

def center(wave):
    """Centers the wave based on the mean"""
    wave = wave - wave.mean()
    return wave

def taper(wave, taper_fraction=0.02):
    taper_length = int(len(wave) * taper_fraction)
    # Build the start tapper
    taper_range = np.arange(taper_length)
    taper_start = np.sin(taper_range/taper_length*np.pi/2)
    # Build the end tapper
    taper_end = np.sin(np.pi/2 + taper_range/taper_length*np.pi/2)
    # Build a center section of only 1s
    taper_center = np.ones(len(wave)-2*taper_length)
    # Concatenate the start, center, and end
    taper_function = np.concatenate([taper_start,taper_center,taper_end])
    # Multiply the wave by the taper function
    wave = wave * taper_function
    return wave

def highpass_filter(wave, highpass_cutoff=1, sampling_rate=20, highpass_order=4):
    """ High pass filter using a butter-lowpass.
    The cutoff and sampling_rate parameters are in Hz.
    The order dictates the attenuation after the cutoff frequency.
    A low order has a long attenuation.
    Ref website: https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units"""

    nyq = 0.5 * sampling_rate
    normal_cutoff = highpass_cutoff / nyq
    b, a = butter(highpass_order, normal_cutoff, btype='high', analog=False)

    wave = lfilter(b, a, wave)

    return wave

def lowpass_filter(wave, lowpass_cutoff=5, sampling_rate=20, lowpass_order=4):
    """ Low pass filter using a butter-lowpass.
    The cutoff and sampling_rate parameters are in Hz.
    The order dictates the attenuation after the cutoff frequency.
    A low order has a long attenuation."""

    nyq = 0.5 * sampling_rate
    normal_cutoff = lowpass_cutoff / nyq
    b, a = butter(lowpass_order, normal_cutoff, btype='low', analog=False)

    wave = lfilter(b, a, wave)

    return wave

def soft_clip(data):
    """Changed to just normalize the data"""
    data = data/(max(data))
    return data

def scale_by_varience(data, decimals=3):
    """Reduces sample by deviding by the varience of every wave"""
    data = data.copy()

    for idx, observation in enumerate(data):
        channel0 = observation[0]
        channel0 = channel0/(np.var(channel0)**0.5)
        data[idx][0] = channel0
        
        channel1 = observation[1]
        channel1 = channel1/(np.var(channel1)**0.5)
        data[idx][1] = channel1
        
        channel2 = observation[2]
        channel2 = channel2/(np.var(channel2)**0.5)
        data[idx][2] = channel2

    data = np.around(data,decimals=decimals)
    
    return data   

def shuffle_data(dataset, labels):
    """Shuffles the dataset and labels to randomize them for training"""
    indexes = np.arange(len(labels))
    np.random.shuffle(indexes) # in this case inplace=True

    # Make holding lists
    samples_shuffled = []
    labels_shuffled = []

    for index_shuffled in indexes:
        wave = dataset[index_shuffled]
        label = labels[index_shuffled]

        samples_shuffled.append(wave)
        labels_shuffled.append(label)

    samples_shuffled = np.array(samples_shuffled)
    labels_shuffled = np.array(labels_shuffled)

    return samples_shuffled, labels_shuffled

def reshape_data(dataset):
    """Takes dataset of (n_observations, n_channels, n_readings)
    and returns (n_observations, n_readings, n_channels)"""
    reshape_sample = np.array([sample.T for sample in dataset])
    return reshape_sample