import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from src.physic_model import Helicopter
from src.surrogate_model import SurrogateModel, SurrogateOptimizer
from src.visualization import DataVisualization
from src.create_dataset import Sample, Dataset

design_space = {
    "rho": [1.1, 1.125],
    "f_factor": [0.2, 1.5],
    "weight": [5000, 50000],
    "k_factor": [1.1, 1.5],
    "rotor_radius": [3.5, 10],
    "velocity": [20, 200]
}


st.set_page_config(page_title="Helicopter Drag Analysis", layout="wide")

st.title("🚁 Helicopter Drag Analysis Dashboard")
st.markdown("""
This application uses a **Surrogate AI Model** to predict helicopter drag in real-time. 
Adjust the sliders on the left to see how parameters affect the optimal velocity.
""")

# --- QUẢN LÝ MODEL (CACHE) ---
@st.cache_resource # Dùng cache để không phải load model mỗi lần kéo slider
def load_surrogate_model():
    model_file = 'data/helicopter_surrogate.pkl'
    trainer = SurrogateModel(pd.DataFrame())
    
    if os.path.exists(model_file):
        success = trainer.loadModel(model_file)
    else:
        success = False

    if not success:
        # Nếu chưa có model, tiến hành train mới (chỉ chạy 1 lần duy nhất)
        sample_engine = Sample(design_space, 6000)
        dataset_engine = Dataset(sample_engine)
        df = dataset_engine.CreateDataset()
        trainer = SurrogateModel(df)
        trainer.train(degree=3)
        trainer.saveModel(model_file)
    
    return trainer

trainer = load_surrogate_model()

# --- SIDEBAR: USER CONFIG (SLIDERS) ---
st.sidebar.header("Input Parameters")

rho = st.sidebar.slider("Air Density (rho) [kg/m3]", 1.1, 1.125, 1.112, step=0.001)
f_factor = st.sidebar.slider("Equivalent Flat Plate Area (f) [m2]", 0.2, 1.5, 1.0)
weight = st.sidebar.slider("Aircraft Weight (W) [N]", 5000, 50000, 30000)
k_factor = st.sidebar.slider("Induced Drag Factor (k)", 1.1, 1.5, 1.3)
rotor_radius = st.sidebar.slider("Rotor Radius (R) [m]", 3.5, 10.0, 5.0)


v_min, v_max = st.sidebar.select_slider(
    "Velocity Range for Analysis [m/s]",
    options=list(range(10, 251, 5)),
    value=(20, 200)
)


user_config = {
    "rho": rho,
    "f_factor": f_factor,
    "weight": weight,
    "k_factor": k_factor,
    "rotor_radius": rotor_radius,
    "velocity": [v_min, v_max]
}

col1, col2 = st.columns([3, 1])

with col1:
    optimizer = SurrogateOptimizer(trainer, user_config)
    plot_engine = DataVisualization()
    

    plot_engine.plotUserConfig(optimizer)
    st.pyplot(plt.gcf()) 

with col2:
    st.subheader("Model Performance")
    m = trainer.performance_metrics
    if m:
        st.metric("R² Score", f"{m['r2']:.4f}")
        st.metric("MAE", f"{m['mae']:.2f} N")
        st.metric("MAPE", f"{m['mape']:.2f}%")
    

    v_range, theory, ai, v_opt_t, v_opt_a, d_min_t, d_min_a = optimizer.compare()
    st.subheader("Optimization Results")
    st.write(f"**Optimal Velocity (AI):** {v_opt_a:.2f} m/s")
    st.write(f"**Minimum Drag (AI):** {d_min_a:.2f} N")