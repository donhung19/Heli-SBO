import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import os

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

def main():
    model_file = 'data/helicopter_surrogate.pkl'
    trainer = SurrogateModel(pd.DataFrame())

    success = False
    if os.path.exists(model_file):
        # This updates trainer.model, trainer.features, etc. internally
        success = trainer.loadModel(model_file)

    # If loading failed or file didn't exist, train a new one
    if not success:
        print("Model not found or incompatible. Training new model...")
        sample_engine = Sample(design_space, 6000)
        dataset_engine = Dataset(sample_engine)
        df = dataset_engine.CreateDataset()
        
        trainer = SurrogateModel(df) # Re-init with real data
        trainer.train(degree=3)
        trainer.saveModel(model_file)
    
    trainer.validateModel()

    # Check coverage of dataset
    # check_coverage(df)

    # Information of validate model
    
    # User config
    user_config = {
        "rho": 1.125,
        "f_factor": 1,
        "weight": 30000,
        "k_factor": 1.5,
        "rotor_radius": 5,
        "velocity": [100, 200]
    }

    plot = DataVisualization()
    optimizer = SurrogateOptimizer(trainer, user_config)
    plot.plotUserConfig(optimizer)


if __name__ == "__main__":
    main()