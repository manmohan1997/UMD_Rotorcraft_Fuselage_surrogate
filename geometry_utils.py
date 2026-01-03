import numpy as np
import trimesh
import meshio
import os
from scipy.optimize import curve_fit

def extract_points(file_path, num_points=100000):
    if file_path.endswith('.stl'):
        mesh = trimesh.load(file_path, force='mesh')
        points = mesh.sample(num_points)
    elif file_path.endswith('.vtu'):
        mesh_data = meshio.read(file_path)
        points = mesh_data.points
    
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    scale = 2 / (x_max - x_min)
    points[:, 0] = 2 * (points[:, 0] - x_min) / (x_max - x_min)
    points[:, 1] *= scale
    points[:, 2] *= scale
    return points

def superellipse_fit(y, a, b, N):
    y_abs = np.abs(y)
    inside = np.maximum(1 - (y_abs / a) ** N, 0)
    return b * inside ** (1.0 / N)

def compute_params(points, num_sections=200):
    points = points[np.argsort(points[:, 0])]
    x_sections = np.linspace(0, 2, num_sections)
    tolerance = 2.0 / num_sections
    
    w_list, h_list, c_list, p_list, x_list = [], [], [], [], []
    for x in x_sections:
        section = points[np.abs(points[:, 0] - x) < tolerance / 2]
        if len(section) < 6: continue
        y_v, z_v = section[:, 1], section[:, 2]
        w, h = y_v.max() - y_v.min(), z_v.max() - z_v.min()
        c = (z_v.max() + z_v.min()) / 2
        try:
            y_f = np.abs(y_v - ((y_v.max() + y_v.min())/2))
            z_f = np.abs(z_v - c)
            popt, _ = curve_fit(superellipse_fit, y_f, z_f, p0=[w/2, h/2, 3], bounds=([1e-3, 1e-3, 2], [5, 5, 10]))
            N_val = popt[2]
        except: N_val = 3.0
        w_list.append(w); h_list.append(h); c_list.append(c); p_list.append(N_val); x_list.append(x)
    return np.array(x_list), np.array(w_list), np.array(h_list), np.array(c_list), np.array(p_list)