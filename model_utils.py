import numpy as np
import tensorflow as tf
import joblib
from scipy.ndimage import convolve1d
def reconstruct_polars(modes_pred, assets):
    results = {}
    curr = 0
    for c in ['CL', 'CD', 'CM']:
        n = assets[c]['num_modes']
        recon = (modes_pred[:, curr:curr+n] @ assets[c]['basis_vectors'].T) + assets[c]['mean_vector']
        results[c] = recon.flatten()
        curr += n
    return results

def get_65_vector(x_uv, Z0, H0, W0, N0, fd_Z, fd_H, fd_W, fd_N, offset_LE, offset_TE, cp_indices):
    indices = [0] + sorted(cp_indices) + [len(x_uv)-1]
    table = []
    for cp in indices:
        x_val = 0.0 if cp == 0 else (2.0 if cp == len(x_uv)-1 else x_uv[cp])
        table.append([x_val, Z0[cp], H0[cp], W0[cp], N0[cp], fd_Z[cp], fd_H[cp], fd_W[cp], fd_N[cp], offset_LE, offset_TE])
    return np.array(table).flatten().reshape(1, -1)

