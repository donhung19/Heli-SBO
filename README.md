# Heli-SBO: Helicopter Surrogate-Based Optimization

**Heli-SBO** is a specialized framework designed to accelerate helicopter aerodynamic analysis. By combining high-fidelity physics models with Machine Learning, it provides real-time drag prediction and cruise velocity optimization through a Surrogate-Based Optimization (SBO) approach.


## 🎯 Project Objectives

* **Drag Prediction**: Accelerate aerodynamic analysis by replacing physics-based formulas with an AI Surrogate Model (reducing computation time from seconds to milliseconds).
* **Cruise Optimization**: Automatically identify the optimal cruise velocity ($V_{opt}$) for any aircraft configuration.
* **Sensitivity Analysis**: Interactive visualization of how aircraft weight, air density, and rotor geometry impact total drag.

## 🛠 Tech Stack

* **Language**: Python 3.13+
* **ML & Analytics**: Scikit-learn (Polynomial Regression, Pipelines), Pandas, NumPy.
* **Sampling**: SciPy (Latin Hypercube Sampling for optimal design space coverage).
* **UI/Deployment**: Streamlit Dashboard for real-time interaction.

## 🚀 Key Features

### 1. High-Fidelity Data Generation
* **Design of Experiments (DoE):** Implements **Latin Hypercube Sampling (LHS)** via `scipy.stats.qmc` to efficiently explore the 6-dimensional design space (Density, Weight, Rotor Radius, etc.).
* **Optimized Sampling:** Ensures a near-random yet structured distribution of 6,000 samples, far superior to standard random sampling for training surrogate models.

### 2. Physics-Informed Surrogate Modeling
* **Custom Feature Engineering:** Enhances the model with physical insights, specifically the inverse quadratic relationship ($1/V^2$) crucial for capturing **Induced Drag** physics.
* **Non-linear Learning:** Uses a Polynomial Pipeline with **Log-Transformation** on the target variable to stabilize variance and achieve an R² score > 0.99.

### 3. Real-Time Optimization (SBO)
* **Instantaneous Optimization:** Replaces iterative physical solvers with a direct AI-driven search, identifying the **Optimal Cruise Velocity ($V_{opt}$)** in milliseconds.
* **Comparative Analysis:** Provides side-by-side comparisons between analytical physics formulas and AI predictions to validate model reliability.

### 4. Interactive Sensitivity Dashboard
* **Dynamic Parameter Tuning:** Built with **Streamlit**, allowing users to manipulate aircraft configurations via sliders and witness real-time shifts in the drag polar curve.
* **Explainable AI:** Enables "What-If" analysis to understand how changes in weight or rotor geometry affect aerodynamic efficiency.

### 5. Robust Persistence Layer
* **Automated Training Workflow:** Features a smart logic that detects missing models, auto-trains on the fly, and persists the entire pipeline (scaler + poly + regressor) using `joblib`.

## 📂 Project Structure
```text
Heli-SBO/
├── app.py                # Streamlit Web Application
├── main.py               # CLI entry point for training and validation
├── data/                 # Directory for stored .pkl models
└── src/
    ├── physic_model.py     # Analytical helicopter drag formulas
    ├── surrogate_model.py  # ML training and optimization logic
    ├── create_dataset.py   # LHS sampling and dataset generation
    └── visualization.py    # Plotting and graph generation

## 🚀 Quick Start

1.  **Install dependencies**: 
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the dashboard**: 
    ```bash
    streamlit run app.py
    ```

## 📊 Performance Summary

* **Accuracy (R² Score)**: > 0.99
* **Error (MAPE)**: < 1%
* **Inference Speed**: Instantaneous feedback upon parameter adjustment.