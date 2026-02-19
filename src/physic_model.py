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

class Helicopter:
    def __init__(self, data):
        self.rho = np.array(data.get('rho'))
        self.f_factor = np.array(data.get('f_factor'))
        self.weight = np.array(data.get('weight'))
        self.k_factor = np.array(data.get('k_factor'))
        self.rotor_radius = np.array(data.get('rotor_radius'))
        self.velocity = np.array(data.get("velocity"))

    def CalculateDrag(self):
        Dp = 0.5 * self.rho * self.velocity**2 * self.f_factor
        Di = (self.k_factor * self.weight**2) / (2 * self.rho * self.velocity**2 * (np.pi) * self.rotor_radius**2)
        return Dp, Di
    
    def FindOptimalVelocity(self, Dt):
        index_opt = np.argmin(Dt)
        v_opt = self.velocity[index_opt]
        D_min = Dt[index_opt]
        return v_opt, D_min