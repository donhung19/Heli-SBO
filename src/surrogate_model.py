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
import joblib

from src.physic_model import Helicopter 

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

class SurrogateModel:
    def __init__(self, dataset):
        # 1. Luôn khởi tạo danh sách features mặc định để tránh lỗi AttributeError
        self.features = ['rho', 'f_factor', 'weight', 'k_factor', 'rotor_radius', 'velocity', 'inv_V2']
        self.model = None
        self.target = 'Total_drag'
        self.performance_metrics = None

        # 2. Xử lý dataset
        if dataset is not None and not dataset.empty:
            self.df = dataset.copy()
            self.df['inv_V2'] = 1 / (self.df['velocity']**2)
        else:
            self.df = dataset

    def train(self, degree):
        if self.df is None or self.df.empty:
            raise ValueError("Không có dữ liệu để huấn luyện!")

        X = self.df[self.features]
        y_log = np.log(self.df[self.target])

        X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', LinearRegression())
        ])
        
        self.model.fit(X_train, y_train_log)

        # Dự báo và tính toán metrics
        y_pred_log = self.model.predict(X_test)
        y_pred = np.exp(y_pred_log)
        y_test_original = np.exp(y_test_log)

        self.performance_metrics = {
            'degree': degree,
            'r2': r2_score(y_test_original, y_pred),
            'mae': mean_absolute_error(y_test_original, y_pred),
            'mape': np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
        }

        return X_test, y_test_original, y_pred
    
    def validateModel(self):
        # Luôn in ra Terminal nếu metrics tồn tại
        if self.performance_metrics is None:
            print("--- Warning: No performance metrics found. Please train the model first. ---")
            return

        m = self.performance_metrics
        print(f"\n" + "="*40)
        print(f"   SURROGATE MODEL PERFORMANCE")
        print(f"="*40)
        print(f"Algorithm: Poly-Regression (Degree {m['degree']})")
        print(f"R2 Score : {m['r2']:.6f}")
        print(f"MAE      : {m['mae']:.2f} Newton")
        print(f"MAPE     : {m['mape']:.4f}%")
        print(f"="*40 + "\n")

    def saveModel(self, path_file):
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(path_file), exist_ok=True)
        
        if self.model is not None:
            model_data = {
                'pipeline': self.model,
                'features': self.features,
                'performance_metrics': self.performance_metrics
            }
            joblib.dump(model_data, path_file)
            print(f"Successfully saved model to: {path_file}")
        else:
            print("Error: Model has not been trained yet!")

    def loadModel(self, path_file):
        try:
            data = joblib.load(path_file)
            self.model = data['pipeline']
            self.features = data['features']
            self.performance_metrics = data['performance_metrics']
            return True  # <--- It returns a single Boolean!
        except:
            return False # <--- It returns a single Boolean!
            
class SurrogateOptimizer:
    def __init__(self, surrogate_trainer, user_config):
        self.cfg = user_config
        self.surrogate = surrogate_trainer

    def compare(self):
        v_min, v_max = self.cfg['velocity']
        v_range = np.linspace(v_min, v_max, 500)

        # Calculating by Theory
        theory_cfg = self.cfg.copy()
        theory_cfg['velocity'] = v_range
        heli_obj = Helicopter(theory_cfg)
        Dp_theory, Di_theory = heli_obj.CalculateDrag()
        total_drag_theory = Dp_theory + Di_theory
        v_opt_theory, d_min_theory = heli_obj.FindOptimalVelocity(total_drag_theory)

        # Surrogate model
        ai_input = pd.DataFrame([self.cfg]*500)
        ai_input['velocity'] = v_range
        ai_input['inv_V2'] = 1 / (v_range**2)

        print(ai_input.columns)

        ai_input = ai_input[self.surrogate.features]
        y_pred_log = self.surrogate.model.predict(ai_input)
        total_drag_ai = np.exp(y_pred_log)

        idx_opt_ai = np.argmin(total_drag_ai)
        v_opt_ai = v_range[idx_opt_ai]
        d_min_ai = total_drag_ai[idx_opt_ai]

        return v_range, total_drag_theory, total_drag_ai, v_opt_theory, v_opt_ai, d_min_theory, d_min_ai
    