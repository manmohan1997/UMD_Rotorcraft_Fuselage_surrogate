import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.chebyshev import Chebyshev
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.linalg import lstsq

def forward_fill(arr):
        """Replaces NaN values with the previous non-NaN value."""
        for i in range(1, len(arr)):
            if np.isnan(arr[i]):
                arr[i] = arr[i - 1]
                # if i == 0:
                #     arr[i] = arr[i +1]
                # else:
                #     arr[i] = arr[i - 1]
        return arr



def compute_hermite_segments(x_values, y_values, slp, segments, is_c1_continuous=True):
    """
    Computes cubic Hermite coefficients for each segment.
    If is_c1_continuous is True, a single global optimization (with weighted residuals)
    is used to maintain both C0 and C1 continuity; otherwise, standard least
    squares fitting is performed on each segment individually.
    
    Parameters:
        x_values (list or np.ndarray): x-coordinates of the data points.
        y_values (list or np.ndarray): y-coordinates of the data points.
        slp (list or np.ndarray): Slopes at each point (first derivatives).
        segments (list of tuples): Each tuple (start_idx, end_idx) defines a segment.
        is_c1_continuous (bool): Flag indicating whether to enforce C1 continuity globally.
        
    Returns:
        list: List of tuples with each tuple containing the segment domain ([x_start, x_end])
              and the optimized cubic Hermite coefficients ([c3, c2, c1, c0]).
    """
    # Ensure inputs are numpy arrays and forward-fill any missing values
    y_values = np.array(y_values, dtype=np.float64)
    slp = np.array(slp, dtype=np.float64)
    y_values = forward_fill(y_values)
    slp = forward_fill(slp)

    optimized_segments = []

    if is_c1_continuous:
        num_segments = len(segments)
        global_initial_coeffs = []
        
        for start_idx, end_idx in segments:
            x0 = x_values[start_idx]
            x1 = x_values[end_idx]
            dx = x1 - x0

            if np.isclose(dx, 0):
                raise ValueError(f"Invalid segment ({start_idx}, {end_idx}): dx is zero.")

            dy = y_values[end_idx] - y_values[start_idx]
            m_start = slp[start_idx]
            m_end = slp[end_idx]

            # Ensure valid coefficients
            c3 = (m_end + m_start - 2 * (dy / dx)) / (dx**2) if dx != 0 else 0
            c2 = (3 * (dy / dx) - 2 * m_start - m_end) / dx if dx != 0 else 0
            c1 = m_start
            c0 = y_values[start_idx]
            global_initial_coeffs.extend([c3, c2, c1, c0])
        global_initial_coeffs = np.array(global_initial_coeffs, dtype=np.float64)
        
        # Define residual weights.
        # w_data scales the residuals from the fitting errors on the given data points.
        # w_cont scales the residuals enforcing C0 (position) and C1 (slope) continuity.
        w_data = 1.0
        w_cont = 10.0  # (You may adjust this value based on the relative scales)

        def global_residuals(coeffs):
            res = []
            for k, (start_idx, end_idx) in enumerate(segments):
                # Get current segment domain and coefficients.
                x0 = x_values[start_idx]
                x1 = x_values[end_idx]
                dx = x1 - x0
                c3, c2, c1, c0 = coeffs[4 * k: 4 * k + 4]
                
                # Data fitting: Evaluate the polynomial at all points within the segment.
                x_seg = x_values[start_idx: end_idx + 1]
                y_seg = y_values[start_idx: end_idx + 1]
                x_rel = x_seg - x0  # Parameterize each segment from 0 to dx.
                y_fit = c3 * (x_rel ** 3) + c2 * (x_rel ** 2) + c1 * x_rel + c0
                res.extend(w_data * (y_fit - y_seg))
                
                # For segments after the first, enforce continuity.
                if k > 0:
                    # Retrieve previous segment’s data 
                    prev_start_idx, prev_end_idx = segments[k - 1]
                    x0_prev = x_values[prev_start_idx]
                    x1_prev = x_values[prev_end_idx]
                    dx_prev = x1_prev - x0_prev
                    c3_prev, c2_prev, c1_prev, c0_prev = coeffs[4 * (k - 1): 4 * (k - 1) + 4]
                    
                    # End of the previous segment
                    y_prev_end = c3_prev * (dx_prev ** 3) + c2_prev * (dx_prev ** 2) + c1_prev * dx_prev + c0_prev
                    slope_prev = 3 * c3_prev * (dx_prev ** 2) + 2 * c2_prev * dx_prev + c1_prev
                    
                    # Beginning of the current segment (at x_rel = 0)
                    y_curr_start = c0
                    slope_curr_start = c1
                    
                    # Enforce C0 and C1 continuity using weighted residuals.
                    res.append(w_cont * (y_curr_start - y_prev_end))
                    res.append(w_cont * (slope_curr_start - slope_prev))
            return np.array(res)
        
        # No hard bounds are enforced; we use -∞ to ∞.
        bounds = ([-np.inf] * len(global_initial_coeffs), [np.inf] * len(global_initial_coeffs))
        result = least_squares(global_residuals, global_initial_coeffs, bounds=bounds)
        global_coeffs_optimized = result.x
        
        # Split the global coefficient vector back into segments.
        for k, (start_idx, end_idx) in enumerate(segments):
            x_start = x_values[start_idx]
            x_end = x_values[end_idx]
            coeffs_segment = global_coeffs_optimized[4 * k: 4 * k + 4]
            optimized_segments.append(([x_start, x_end], coeffs_segment))
    else:
        # Perform standard (local) least squares fitting for each segment separately.
        def lsq_residuals(coeffs, x_seg, y_seg):
            x_rel = x_seg - x_seg[0]
            c3, c2, c1, c0 = coeffs
            return c3 * (x_rel ** 3) + c2 * (x_rel ** 2) + c1 * x_rel + c0 - y_seg

        for start_idx, end_idx in segments:
            x0 = x_values[start_idx]
            x1 = x_values[end_idx]
            dx = x1 - x0
            dy = y_values[end_idx] - y_values[start_idx]
            m_start = slp[start_idx]
            m_end = slp[end_idx]
            # Initial guess using cubic Hermite formulas.
            c3 = (m_end + m_start - 2 * (dy / dx)) / (dx**2)
            c2 = (3 * (dy / dx) - 2 * m_start - m_end) / dx
            c1 = m_start
            c0 = y_values[start_idx]
            initial_coeffs = [c3, c2, c1, c0]
            
            x_seg = x_values[start_idx: end_idx + 1]
            y_seg = y_values[start_idx: end_idx + 1]
            result = least_squares(lsq_residuals, initial_coeffs, args=(x_seg, y_seg))
            coeffs_optimized = result.x
            optimized_segments.append(([x0, x1], coeffs_optimized))
            
    return optimized_segments




# fitting segments maintaining C0 and C1 continuty 
def evaluate_hermite_segments(x_values_new, fitted_segments_with_domain):
    """Evaluates y-values for new x-values using the fitted Hermite segments.

    Parameters:
        x_values_new (list or np.ndarray): New x-values where the y-values need to be predicted.
        fitted_segments_with_domain (list): List of tuples, each containing the segment's domain and the optimized coefficients.

    Returns:
        np.ndarray: Predicted y-values for the new x-values.
    """
    y_values_new = np.zeros_like(x_values_new)

    for i, x_new in enumerate(x_values_new):
        for j, (domain, coefficients) in enumerate(fitted_segments_with_domain):
            x_start, x_end = domain
            if x_start <= x_new <= x_end:
                # Compute dx and evaluate the polynomial
                dx = x_new - x_start
                c3, c2, c1, c0 = coefficients
                y_values_new[i] = c3 * dx**3 + c2 * dx**2 + c1 * dx + c0
                break

    return y_values_new




#chevyshev Method fititng
def compute_chebyshev_segments(x_values, y_values, segments, degree=3):
    fitted_segments = []

    for start_idx, end_idx in segments:
        x_segment = x_values[start_idx:end_idx+1]
        y_segment = y_values[start_idx:end_idx+1]

        if len(x_segment) > degree:
            cheb_fit = Chebyshev.fit(x_segment, y_segment, degree)
            fitted_segments.append(cheb_fit)
        else:
            fitted_segments.append(None)  # Not enough points for the polynomial degree

    return fitted_segments

def evaluate_chebyshev_segments(x_values_new, x_values, fitted_segments, segments):
    y_values_new = np.zeros_like(x_values_new)

    for i, x_new in enumerate(x_values_new):
        for j, (start_idx, end_idx) in enumerate(segments):
            if x_values[start_idx] <= x_new <= x_values[end_idx]:
                cheb_fit = fitted_segments[j]
                if cheb_fit is not None:
                    y_values_new[i] = cheb_fit(x_new)
                else:
                    y_values_new[i] = np.nan  # Use NaN to indicate invalid segments
                break

    return y_values_new


def CST_varriable_conversion(x_vals, heights, widths, cambers, powers, 
                                          x_start=0.01, x_end=1.9):
    """
    Extracts and normalizes fuselage shape functions between x_start and x_end.
    Camber is offset using smooth blending of LE and TE values.

    Parameters:
        x_vals: np.ndarray - original x positions (cosine spaced)
        heights, widths, cambers, powers: np.ndarray - shape parameters
        x_start, x_end: float - trim limits in x for valid geometry

    Returns:
        x_values, H_0, W_0, Z0_0, N_0
    """
    offset_LE = cambers[0]
    offset_TE = cambers[-1]

    index_to_slice = np.argmin(np.abs(x_vals - x_start))
    last_index = np.argmin(np.abs(x_vals - x_end))

    x_values = x_vals[index_to_slice:last_index+1]
    H_0 = heights[index_to_slice:last_index+1]
    W_0 = widths[index_to_slice:last_index+1]
    Z0_0 = cambers[index_to_slice:last_index+1]
    N_0 = powers[index_to_slice:last_index+1]

    # Correction functions
    CF_Height = (x_values / 2)**0.555 * (1 - x_values / 2)**0.5
    CF_Width  = (x_values / 2)**0.5   * (1 - x_values / 2)**0.5
    CF_Camber = (x_values / 2)**0.555 * (1 - x_values / 2)

    # Normalize H and W
    H_0 = H_0 / CF_Height
    W_0 = W_0 / CF_Width

    # Normalize camber using LE/TE offset removal
    Z0_0 = (Z0_0 - offset_LE * (1 - x_values / 2) - offset_TE * (x_values / 2)) / CF_Camber

    return x_values, H_0, W_0, Z0_0, N_0
