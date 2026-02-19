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


class DataVisualization:
    def plotValidateModel(self, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.3, color='teal', s=10)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Theory (Newton)")
        plt.ylabel("Surrogate model (Newton)")
        plt.title("Plot Parity: Comparsion of Theory and Surrogate model")
        plt.grid(True)
        plt.show()

    def plotUserConfig(self, obj_SurrogateOptimizer):
        v_range, total_drag_theory, total_drag_ai, v_opt_theory, v_opt_ai, d_min_theory, d_min_ai = obj_SurrogateOptimizer.compare()

        plt.figure(figsize=(11, 7))
        plt.plot(v_range, total_drag_theory, 'k-', lw=3, label='Theoretical (Analytical)', alpha=0.5)
        plt.plot(v_range, total_drag_ai, 'r--', lw=2, label='Predicted (Surrogate Model)')
        
        # Mark Optimal Points
        plt.scatter(v_opt_theory, d_min_theory, color='black', s=100, 
                    label=f'Theory: V={v_opt_theory:.2f} m/s, D={d_min_theory:.2f} N')
        plt.scatter(v_opt_ai, d_min_ai, color='red', marker='x', s=120, 
                    label=f'AI: V={v_opt_ai:.2f} m/s, D={d_min_ai:.2f} N')

        plt.title("Total Drag Curve Comparison: Theory vs. AI", fontsize=14)
        plt.xlabel("Velocity (m/s)", fontsize=12)
        plt.ylabel("Total Drag (Newton)", fontsize=12)
        
        plt.legend(loc='upper right', frameon=True, shadow=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
#        plt.show()