import numpy as np
from scipy.signal import argrelextrema


def find_critical_points(f_prime, f_double_prime):
    """
    Find critical points of a curve based on the maxima and minima of the first and second derivatives.

    Parameters:
        f_prime (array): First derivative values.
        f_double_prime (array): Second derivative values.

    Returns:
        tuple:
            - critical_indices (array): Combined critical points from maxima/minima of first and second derivatives.
            - extrema_indices_first (array): Critical points from maxima/minima of the first derivative.
            - extrema_indices_second (array): Critical points from maxima/minima of the second derivative.
    """
    # Find local maxima and minima of the first derivative
    maxima_indices_first = argrelextrema(f_prime, np.greater)[0]
    minima_indices_first = argrelextrema(f_prime, np.less)[0]
    extrema_indices_first = np.sort(np.append(maxima_indices_first, minima_indices_first))

    # Find local maxima and minima of the second derivative
    maxima_indices_second = argrelextrema(f_double_prime, np.greater)[0]
    minima_indices_second = argrelextrema(f_double_prime, np.less)[0]
    extrema_indices_second = np.sort(np.append(maxima_indices_second, minima_indices_second))

    # Combine all critical point indices
    critical_indices = np.sort(np.unique(np.append(extrema_indices_first, extrema_indices_second)))

    return critical_indices, extrema_indices_first, extrema_indices_second

# def create_segments(crit_points, x_values_length):
#     comb_indices = [0] + crit_points + [x_values_length - 1]
#     sgmnts = [(comb_indices[i], comb_indices[i + 1]) for i in range(len(comb_indices) - 1)]
#     return sgmnts

def create_segments(critical_points, x_values_length):
    # Ensure critical_points is a list of integers
    if isinstance(critical_points, np.ndarray):
        critical_points = critical_points.tolist()
    else:
        critical_points = list(map(int, critical_points))   
    # Combine the indices properly
    combined_indices = [0] + critical_points + [x_values_length - 1]
    segments = [(combined_indices[i], combined_indices[i + 1]) for i in range(len(combined_indices) - 1)]
    return segments