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

design_space = {
    "rho": [1.1, 1.125],
    "f_factor": [0.2, 1.5],
    "weight": [5000, 50000],
    "k_factor": [1.1, 1.5],
    "rotor_radius": [3.5, 10],
    "velocity": [20, 200]
}

# Tách hàm này ra ngoài để dễ sử dụng
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

class SurrogateModel:
    def __init__(self, dataset):
        self.df = dataset.copy()
        self.model = None
        self.df['inv_V2'] = 1 / (self.df['velocity']**2)
        self.features = ['rho', 'f_factor', 'weight', 'k_factor', 'rotor_radius', 'velocity','inv_V2']
        self.target = 'Total_drag'

    def train(self, degree):
        X = self.df[self.features]
        # LOG-TRANSFORM target
        y_log = np.log(self.df[self.target])

        X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', LinearRegression())
        ])
        
        self.model.fit(X_train, y_train_log)

        # Dự báo trên thang log
        y_pred_log = self.model.predict(X_test)
        
        # CHUYỂN NGƯỢC VỀ NEWTON ĐỂ TÍNH SAI SỐ
        y_pred = np.exp(y_pred_log)
        y_test_original = np.exp(y_test_log)

        r2 = r2_score(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100

        print(f"--- Complete training !!! (Degree {degree} + Log-Transform) ---")
        print(f"R2 Score: {r2:.6f}")
        print(f"MAE: {mae:.2f} Newton")
        print(f"MAPE: {mape:.4f}%")
        return X_test, y_test_original, y_pred

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
        ai_input = pd.DataFrame([self.cfg]*500) #why is 500 ???
        ai_input['velocity'] = v_range
        ai_input['inv_V2'] = 1 / (v_range**2)

        ai_input = ai_input[self.surrogate.features]
        y_pred_log = self.surrogate.model.predict(ai_input)
        total_drag_ai = np.exp(y_pred_log)

        idx_opt_ai = np.argmin(total_drag_ai)
        v_opt_ai = v_range[idx_opt_ai]
        d_min_ai = total_drag_ai[idx_opt_ai]

        return v_range, total_drag_theory, total_drag_ai, v_opt_theory, v_opt_ai, d_min_theory, d_min_ai
    
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


        plt.figure(figsize=(10, 6))
        plt.plot(v_range, total_drag_theory, 'k-', lw=3, label='Lý thuyết (Công thức)', alpha=0.5)
        plt.plot(v_range, total_drag_ai, 'r--', lw=2, label='Dự báo (Surrogate Model)')
        
        # Đánh dấu các điểm cực tiểu (V_opt)
        plt.scatter(v_opt_theory, d_min_theory, color='black', s=100, 
                    label=f'V_opt Lý thuyết: {v_opt_theory:.2f} m/s')
        plt.scatter(v_opt_ai, d_min_ai, color='red', marker='x', s=120, 
                    label=f'V_opt AI: {v_opt_ai:.2f} m/s')
        
        plt.title("So sánh Đường cong Lực cản: Lý thuyết vs. AI")
        plt.xlabel("Vận tốc (m/s)")
        plt.ylabel("Tổng lực cản (Newton)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

def main():
    sample_engine = Sample(design_space, 6000)
    dataset_engine = Dataset(sample_engine)
    df = dataset_engine.CreateDataset()
    plot = DataVisualization()

    # Check coverage of dataset
    check_coverage(df)

    # Train surrogate model 
    trainer = SurrogateModel(df)
    X_test, y_test, y_pred = trainer.train(3)
    plot.plotValidateModel(y_test, y_pred)

    # User config
    user_config = {
        "rho": 1.125,
        "f_factor": 1,
        "weight": 30000,
        "k_factor": 1.5,
        "rotor_radius": 5,
        "velocity": [20, 200]
    }

    optimizer = SurrogateOptimizer(trainer, user_config)
    plot.plotUserConfig(optimizer)


if __name__ == "__main__":
    main()