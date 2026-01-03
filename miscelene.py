import numpy as np
from scipy.signal import find_peaks
# finding first and second derivative of the camber line to determine critical points
def first_derivative(x, y):
    """Calculate the first derivative dy/dx."""
    # Using finite difference method
    dy_dx = np.gradient(y, x)  # Compute the gradient
    return dy_dx

def second_derivative(x, y):
    """Calculate the second derivative d^2y/dx^2."""
    # First, find the first derivative
    dy_dx = first_derivative(x, y)
    # Then, compute the second derivative
    d2y_dx2 = np.gradient(dy_dx, x)  # Compute the gradient of the first derivative
    return d2y_dx2




def map_values_with_aspect_ratio(x_values, y_values, new_x_min=0, new_x_max=1):
    # Convert input lists to numpy arrays if they aren't already
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Determine the original range of x_values
    old_x_min = x_values.min()
    old_x_max = x_values.max()

    if old_x_min == old_x_max:
        raise ValueError("All x_values are the same; cannot map a single-point range.")

    # Map x_values to the new range [new_x_min, new_x_max]
    mapped_x = new_x_min + (x_values - old_x_min) * (new_x_max - new_x_min) / (old_x_max - old_x_min)

    # Scale y_values proportionally to maintain aspect ratio
    y_scale_factor = (new_x_max - new_x_min) / (old_x_max - old_x_min)
    mapped_y = y_values * y_scale_factor

    return mapped_x, mapped_y


# import numpy as np


def add_points_near_high_curvature(
    x_values, 
    curvature, 
    n_regions=3, 
    points_per_region=10, 
    window_fraction=0.02, 
    manual_regions=None
):
    """
    Adds extra points to x_values in regions of highest curvature or user-defined regions.

    Parameters:
    - x_values: numpy array of original x coordinates
    - curvature: numpy array of same length, curvature values at x_values
    - n_regions: number of peak regions to refine (default 3)
    - points_per_region: number of new points per region (default 10)
    - window_fraction: if auto, sets half-width of region as a fraction of domain length
    - manual_regions: list of (xmin, xmax) tuples. If provided, these are used instead of automatic detection.

    Returns:
    - new_x_values: sorted numpy array with original and extra points
    """

    x_min, x_max = np.min(x_values), np.max(x_values)
    x_extra = []

    if manual_regions is not None:
        # Use manually defined regions
        for xmin, xmax in manual_regions:
            new_points = np.linspace(xmin, xmax, points_per_region)
            new_points = new_points[(new_points >= x_min) & (new_points <= x_max)]
            x_extra.extend(new_points)
    else:
        # Use curvature to identify regions
        peaks, _ = find_peaks(curvature, distance=10)
        if len(peaks) == 0:
            return x_values  # No peaks, return original

        top_indices = peaks[np.argsort(curvature[peaks])[-n_regions:]]
        window_size = (x_max - x_min) * window_fraction

        for idx in top_indices:
            x0 = x_values[idx]
            new_points = np.linspace(x0 - window_size, x0 + window_size, points_per_region)
            new_points = new_points[(new_points >= x_min) & (new_points <= x_max)]
            x_extra.extend(new_points)

    # Combine and sort
    new_x_values = np.unique(np.sort(np.concatenate((x_values, x_extra))))
    return new_x_values



def curvature_from_derivatives(dy, ddy):
    """
    Compute curvature given first and second derivatives.

    Parameters:
    - dy : array-like, first derivative values y'(x)
    - ddy : array-like, second derivative values y''(x)

    Returns:
    - curvature : array-like, curvature κ(x)
    """
    dy = np.asarray(dy)
    ddy = np.asarray(ddy)
    
    curvature = ddy / (1 + dy**2)**1.5
    return curvature


def calculate_fuselage_coordinates_fromCST(x_values, theta_degrees, Z0, H, W, N):
    """
    Calculate fuselage coordinates for a range of x values and specific theta degrees, using given Z0, H, W, and N values.

    Parameters:
        x_values (array-like): The range of x values.
        theta_degrees (array-like): The theta values in degrees for which coordinates are calculated.
        Z0 (array-like): Camber (vertical offset) values corresponding to x_values.
        H (array-like): Height values corresponding to x_values.
        W (array-like): Width values corresponding to x_values.
        N (array-like): N values corresponding to x_values.

    Returns:
        dict: A dictionary with theta values in degrees as keys and (x, y, z) tuples as values.
    """
    fuselage_coordinates = {}

    # Iterate over each theta degree
    for theta_deg in theta_degrees:
        theta_rad = np.deg2rad(theta_deg)  # Convert theta to radians
        temp_x_coords, temp_y_coords, temp_z_coords = [], [], []

        # Iterate over all x values
        for i, x in enumerate(x_values):
            h = H[i]
            w = W[i]
            z0 = Z0[i]
            n = N[i]

            # Calculate the radius using the formula
            sin_theta, cos_theta = np.sin(theta_rad), np.cos(theta_rad)
            denom = (abs((h / 2) * sin_theta)**n + abs((w / 2) * cos_theta)**n)
            denom = np.where(denom == 0, np.spacing(1.0), denom)  # Replace zero values with a very small number


            r = ((h * w / 4)**n / denom)**(1 / n)
            y, z = r * sin_theta, r * cos_theta + z0

            # Append the calculated coordinates
            temp_x_coords.append(x)
            temp_y_coords.append(y)
            temp_z_coords.append(z)

        # Store the coordinates for the current theta
        fuselage_coordinates[theta_deg] = (
            np.array(temp_x_coords),  # x-coordinates
            np.array(temp_y_coords),  # y-coordinates
            np.array(temp_z_coords)   # z-coordinates
        )

    return fuselage_coordinates

