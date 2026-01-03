import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from geometry_utils import extract_points, compute_params
from model_utils import get_65_vector, reconstruct_polars
from miscelene import first_derivative
from GA_breakpoints import genetic_algorithm_optimum_critical_points
from Fitting_Methods import CST_varriable_conversion
from scipy.signal import savgol_filter
import streamlit as st
import os

# --- UMD THEME & NEUTRAL TOP-RIGHT CSS ---
st.markdown("""
    <style>
    /* Main App Background */
    .stApp { background-color: #ffffff; }

    /* Sidebar - Light Yellow (UMD Gold) */
    section[data-testid="stSidebar"] {
        background-color: #FFD200 !important;
        border-right: 3px solid #E21833;
    }

    /* NEUTRALIZE TOP RIGHT RED ELEMENTS */
    /* 1. Hide the red line at the very top */
    header[data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0) !important;
    }
    header[data-testid="stHeader"]::after {
        background-image: none !important;
        background-color: transparent !important;
    }

    /* 2. Change the 'Running/Deploy' status box from Red to Gold/Grey */
    [data-testid="stStatusWidget"] {
        background-color: #f8f9fa !important;
        border: 2px solid #FFD200 !important; /* Gold border instead of red */
        color: #222222 !important;
    }

    /* Center Sidebar Content */
    [data-testid="stSidebarNav"] {
        padding-top: 20px;
    }
    
    /* Ensure sidebar labels are black */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label {
        color: #000000 !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR LOGO CENTERING ---
with st.sidebar:
    if os.path.exists("logo.png"):
        # Use columns to center the image
        # [1, 2, 1] ratio makes the middle column 50% width and centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("logo.png", use_column_width=True)
    
    st.markdown("<h3 style='text-align: center; color: #E21833; margin-top:0;'>AGRC Surrogate</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #000000; font-size: 0.9em;'>Alfred Gessow Rotorcraft Center</p>", unsafe_allow_html=True)
    st.divider()
    
    # File uploader and other controls
    uploaded_file = st.file_uploader("Upload CAD Data (STL/VTU)", type=['stl', 'vtu'])

st.title("🚁 Fuselage Surrogate")
if uploaded_file:
    temp_path = f"data_{uploaded_file.name}"
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())

    # 1. Geometry View (Non-blocking)
    view_space = st.empty()
    
    with st.status("Analyzing Airframe...", expanded=True) as status:
        st.write("🌌 Extracting Point Cloud...")
        pts = extract_points(temp_path)
        with view_space.expander("🔍 3D Geometry Inspection", expanded=True):
            fig3d = go.Figure(data=[go.Scatter3d(x=pts[::15,0], y=pts[::15,1], z=pts[::15,2], mode='markers', marker=dict(size=1.2, color=pts[::15,0], colorscale='IceFire'))])
            fig3d.update_layout(scene=dict(aspectmode='data'), height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig3d, use_container_width=True)

        st.write("📏 Computing Sectionals...")
        x_v, w_v, h_v, c_v, p_v = compute_params(pts)
        window_length = 10 
        polyorder = 3 
        h_v = savgol_filter(h_v, window_length, polyorder)
        w_v = savgol_filter(w_v, window_length, polyorder)
        cambec_vrs = savgol_filter(c_v, window_length, polyorder)
        p_v = savgol_filter(p_v, window_length, polyorder)

        x_uv, H0, W0, Z0, N0 = CST_varriable_conversion(x_v, h_v, w_v, c_v, p_v)
        H0 = savgol_filter(H0, window_length, polyorder)
        W0 = savgol_filter(W0, window_length, polyorder)
        Z0 = savgol_filter(Z0, window_length, polyorder)
        N0 = savgol_filter(N0, window_length, polyorder)
    with st.status("Generating Surrogate...", expanded=True) as status:
        st.write("🧬 Genetic Optimization...")
        y_d = {'Z': Z0, 'H': H0, 'W': W0}
        s_d = {'Z': first_derivative(x_uv, Z0), 'H': first_derivative(x_uv, H0), 'W': first_derivative(x_uv, W0)}
        _, cp_idx = genetic_algorithm_optimum_critical_points(x_uv, y_d, s_d)

        st.write("🤖 POD-NN Inference...")
        X_phys = get_65_vector(x_uv, Z0, H0, W0, N0, s_d['Z'], first_derivative(x_uv, H0), first_derivative(x_uv, W0), first_derivative(x_uv, N0), c_v[0], c_v[-1], cp_idx)
        
        # Prediction
        m_scaler = joblib.load('master_scaler.pkl')
        m_pca = joblib.load('master_pca_model.pkl')
        pod_nn = tf.keras.models.load_model('best_pod_mode_model_tuned_120_random_A9999.h5', compile=False)
        assets = joblib.load('pod_reconstruction_assets_ran_A9999.pkl')
        
        X_inp = joblib.load('pca_input_scaler_120_ran_A9999.pkl').transform(m_pca.transform(m_scaler.transform(X_phys)))
        Y_modes = joblib.load('pod_modes_scaler_120_ran_A9999.pkl').inverse_transform(pod_nn.predict(X_inp, verbose=0))
        res = reconstruct_polars(Y_modes, assets)
        status.update(label="✅ Analysis Complete", state="complete")

    # --- THREE SEPARATE POLAR PLOTS ---
    aoa = np.array([-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0])
    st.subheader("📊 Aerodynamic Performance")
    p1, p2, p3 = st.columns(3)

    with p1:
        f1 = go.Figure(); f1.add_trace(go.Scatter(x=aoa, y=res['CL'], name="CL", line=dict(color='#00d4ff', width=3)))
        f1.update_layout(title="Lift Coefficient (CL)", template="plotly_dark")
        st.plotly_chart(f1, use_container_width=True)
    with p2:
        f2 = go.Figure(); f2.add_trace(go.Scatter(x=aoa, y=res['CD'], name="CD", line=dict(color='#ff007f', width=3)))
        f2.update_layout(title="Drag Coefficient (CD)", template="plotly_dark")
        st.plotly_chart(f2, use_container_width=True)
    with p3:
        f3 = go.Figure(); f3.add_trace(go.Scatter(x=aoa, y=res['CM'], name="CM", line=dict(color='#00ff88', width=3)))
        f3.update_layout(title="Moment Coefficient (CM)", template="plotly_dark")
        st.plotly_chart(f3, use_container_width=True)

    # --- .DAT EXPORT ---
    st.divider()
    dat_content = "AOA\tCL\tCD\tCM\n"
    for i in range(len(aoa)):
        dat_content += f"{aoa[i]:.2f}\t{res['CL'][i]:.6f}\t{res['CD'][i]:.6f}\t{res['CM'][i]:.6f}\n"
    
    st.download_button(label="📥 Download Aerodynamic Data (.dat)", data=dat_content, file_name="aero_results.dat", mime="text/plain")

else:
    st.info("Upload an STL/VTU file to begin processing.")