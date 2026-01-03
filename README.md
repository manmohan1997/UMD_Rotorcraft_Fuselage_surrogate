# UMD Fuselage Surrogate

This is a web-based tool for predicting the aerodynamic coefficients of helicopter fuselages. It was developed at the Alfred Gessow Rotorcraft Center (AGRC) to provide a fast alternative to CFD for initial design phases.

The app takes an STL or VTU file, processes the geometry using a Genetic Algorithm to find key breakpoints, and then uses a trained Neural Network (POD-based) to give you CL, CD, and CM polars.

## How it works
1. **Geometry Processing**: The code extracts cross-sections and fits superellipses to find width, height, and camber.
2. **Breakpoint Optimization**: A Genetic Algorithm (GA) picks 5 critical points along the fuselage that best represent the shape.
3. **Prediction**: These points (plus derivatives and offsets) form a 65-element vector that goes into the model.
4. **Output**: You get three plots (CL, CD, CM) and can download the data as a `.dat` file.

## Files in this Repo
* `app.py`: The main Streamlit script.
* `geometry_utils.py` & `model_utils.py`: Helper functions for CAD and the NN model.
* `GA_breakpoints.py`: The logic for the Genetic Algorithm.
* `Fitting_Methods.py` & `miscelene.py`: Functions for CST conversion and derivatives.
* `*.pkl` & `*.h5`: The trained scalers, PCA, and Neural Network files.

## Local Setup
If you want to run this on your own machine instead of the website:
1. Make sure you have Python 3.9+ installed.
2. Install the requirements:
   ```bash
   pip install streamlit numpy pandas plotly joblib tensorflow scikit-learn trimesh meshio
