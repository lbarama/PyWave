# Author: Jesse Williams
# Company: Global Technology Connection
# Last updated: 2019 11 20

import numpy as np


def energy_ratio(wave, split_ratio=0.5):
    """Calcuates the energy of the first section of the wave and the second.
    Returns the ratio of the second/first. """

    # Find the split point
    n_points = len(wave)
    n_split = int(n_points*split_ratio)
    # Split the wave into two parts
    wave_first = wave[:n_split]
    wave_second = wave[n_split:]
    # Calculate the energy (square of amplitute) and sum it
    energy_first = np.square(wave_first).sum()
    energy_second = np.square(wave_second).sum()
    # Normalize the summation by the number of points
    energy_first = energy_first/len(wave_first)
    energy_second = energy_second/len(wave_second)
    # Energy ratio 
    energy_ratio = energy_second/energy_first
    
    return energy_ratio