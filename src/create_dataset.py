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

from src.physic_model import Helicopter 



def check_coverage(df):
    features = ['rho', 'f_factor', 'weight', 'k_factor', 'rotor_radius', 'velocity']
    sns.set_theme(style="white")
    g = sns.pairplot(df[features], diag_kind='kde', plot_kws={'alpha': 0.4, 's': 2, 'color': 'teal'})
    g.fig.suptitle("Kiểm tra độ bao phủ của 6000 mẫu (LHS)", y=1.02)
    plt.show()

class Sample:
    def __init__(self, design_space, n_samples):
        self.bound = design_space
        self.n_samples = n_samples
        self.labels = list(design_space.keys())
        self.lower_bound = [self.bound[key][0] for key in self.labels]
        self.upper_bound = [self.bound[key][1] for key in self.labels]

    def CreateSample(self):
        sampler = qmc.LatinHypercube(d=len(self.labels))
        sample = sampler.random(n=self.n_samples)
        scaled_sample = qmc.scale(sample, self.lower_bound, self.upper_bound)
        return pd.DataFrame(scaled_sample, columns=self.labels)
    
class Dataset:
    def __init__(self, sample):
        self.sample = sample

    def CreateDataset(self):
        df = self.sample.CreateSample()
        # Chuyển đổi để Helicopter class có thể đọc được
        sample_obj = Helicopter(df.to_dict('list'))

        Dp, Di = sample_obj.CalculateDrag()
        df['Total_drag'] = Dp + Di
        return df