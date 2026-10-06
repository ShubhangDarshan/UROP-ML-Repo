# Wireless Power Transfer (WPT) - Mutual Inductance Machine Learning System

**UROP Research Codebase**  
*Modeling, Prediction, and Estimation of Mutual Inductance ($M$) in High-Frequency Resonant Inductive Wireless Power Transfer Systems using Multiple Perceptron Architectures.*

---

## 1. Project Overview

This research repository develops, trains, and evaluates a **Multiple Perceptron Machine Learning System** (Multilayer Perceptron / Artificial Neural Network) for high-precision prediction of **Mutual Inductance (`Mutual_Inductance_H`)** in Inductive Wireless Power Transfer (WPT) circuits.

In resonant inductive power transfer (such as Series-Series or SS compensation), **Mutual Inductance ($M$)** is the fundamental physical parameter governing magnetic coupling, induced secondary electromotive force (EMF), output voltage, and power transfer efficiency. However, accurately acquiring or tracking mutual inductance during real-time operation or iterative design can be challenging:
- Direct physical measurement typically requires specialized, expensive sensors or offline impedance analyzers.
- Electromagnetic simulation sweeps (FEM / Maxwell / SPICE) are computationally demanding and unsuitable for millisecond-scale closed-loop control or optimization.

This codebase provides an end-to-end Machine Learning pipeline that maps input circuit parameters (and/or measured electrical operating parameters) directly to the predicted value of `Mutual_Inductance_H` with over **99.5% accuracy ($R^2 > 0.995$, $\text{MAPE} < 0.75\%$)** in sub-milliseconds.

---

## 2. Theoretical Background: Mutual Inductance in WPT

### Physical Significance of Mutual Inductance
In a two-coil resonant inductive WPT link, primary and secondary coils are magnetically coupled via mutual inductance $M$:

$$M = k \sqrt{L_1 L_2}$$

Where:
- $L_1$: Transmitter / Primary coil self-inductance ($\text{H}$).
- $L_2$: Receiver / Secondary coil self-inductance ($\text{H}$).
- $k$: Dimensionless magnetic coupling coefficient ($0 \le k \le 1$).

Under harmonic excitation with angular frequency $\omega = 2\pi f$:
- The voltage induced in the secondary coil is: $v_{\text{ind},2}(t) = \omega M I_1$.
- The reflected impedance back to the primary is: $Z_{\text{ref}} = \frac{\omega^2 M^2}{Z_2 + R_L}$.
- The power transferred to the load resistance $R_L$ is strongly dependent on $M^2$:

$$P_{\text{out}} \approx \frac{\omega^2 M^2 I_1^2 R_L}{(R_2 + R_L)^2 + X_2^2}$$

### Machine Learning Multiple Perceptron Formulation
A Multilayer Perceptron maps an $n$-dimensional input feature vector $\mathbf{x}$ to the target output $\hat{y} = \text{Mutual\_Inductance\_H}$ through cascading layers of artificial perceptrons:

$$\mathbf{h}^{(1)} = \sigma\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right)$$

$$\mathbf{h}^{(l)} = \sigma\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l = 2, \dots, L$$

$$\hat{y} = \mathbf{W}^{(L+1)} \mathbf{h}^{(L)} + b^{(L+1)}$$

Where:
- $\mathbf{x}$: Input parameter vector (operating frequency, inductances, resistances, capacitances, voltages, etc.).
- $\mathbf{W}^{(l)}, \mathbf{b}^{(l)}$: Trainable synaptic weight matrices and bias vectors.
- $\sigma(\cdot)$: Non-linear activation function (such as $\text{ReLU}(z) = \max(0, z)$).
- $\hat{y}$: Predicted mutual inductance ($M$) in Henries ($\text{H}$).

---

## 3. Dataset Description (`WPT_1000_Final_Dataset.csv`)

The dataset comprises 1,000 circuit simulations of a Series-Series (SS) resonant inductive power transfer link with varying physical, magnetic, and load parameters.

### Target Output Variable

| Target Name | Description | Physical Unit | Typical Range in Dataset |
| :--- | :--- | :---: | :---: |
| **`Mutual_Inductance_H`** | Mutual inductance between primary and secondary coils ($M$) | $\text{H}$ | $1.29 \times 10^{-5} - 2.51 \times 10^{-5}\,\text{H}$ ($12.9 - 25.1\,\mu\text{H}$) |

### Input Parameters (Configurable Feature Sets)

#### 1. Default Circuit Parameters (10 Features — Recommended)
These represent the core physical and electrical design parameters of the WPT system:

| Feature Name | Description | Physical Unit | Range in Dataset |
| :--- | :--- | :---: | :---: |
| `Frequency_kHz` | Inverter operating switching frequency | $\text{kHz}$ | $80.00 - 89.99$ |
| `L1_H` | Primary coil self-inductance | $\text{H}$ | $6.00 \times 10^{-5} - 8.15 \times 10^{-5}$ |
| `L2_H` | Secondary coil self-inductance | $\text{H}$ | $6.80 \times 10^{-5} - 9.00 \times 10^{-5}$ |
| `k` | Magnetic coupling coefficient | Dimensionless | $0.200 - 0.299$ |
| `R1_Ohm` | Primary coil AC winding resistance | $\Omega$ | $0.040 - 0.080$ |
| `R2_Ohm` | Secondary coil AC winding resistance | $\Omega$ | $0.040 - 0.080$ |
| `RL_Ohm` | Secondary load resistance | $\Omega$ | $15.00 - 50.00$ |
| `C1_F` | Primary series compensation capacitance | $\text{F}$ | $3.70 \times 10^{-8} - 5.99 \times 10^{-8}$ |
| `C2_F` | Secondary series compensation capacitance | $\text{F}$ | $3.52 \times 10^{-8} - 5.75 \times 10^{-8}$ |
| `Vin_RMS_V` | Inverter output RMS excitation voltage | $\text{V}$ | $198.74 - 208.72$ |

*(Note: `ESR_C1_Ohm` and `ESR_C2_Ohm` are constant at $0.01\,\Omega$ throughout the dataset and are automatically identified and dropped during preprocessing).*

#### 2. Extended Feature Set (22 Features)
Includes all above circuit parameters plus measured electrical quantities:
- `Iin_RMS_A`: Primary input current.
- `Vout_RMS_V`: Secondary output voltage across load.
- `Iout_RMS_A`: Secondary load current.
- `Apparent_Power_VA`: System apparent input power.
- `Power_Factor`: Input displacement power factor.
- `Pin_W`: Active input power from source.
- `Pout_W`: Active output power delivered to load.
- `Pcu1_W`, `Pcu2_W`: Copper losses in coils.
- `Power_Balance_W`, `Balance_Error_W`: Conservation of energy metrics.
- `Efficiency_percent`: Overall power transfer efficiency.

#### 3. Sensorless / Measurement Mode (15 Features — Without $k$)
When magnetic coupling $k$ cannot be measured directly in a practical hardware setup, the model can estimate $M$ purely from coil parameters and operating voltages, currents, and powers.

---

## 4. Repository & File Structure

```
D:\UROP Research Code\
│
├── data\
│   └── WPT_1000_Final_Dataset.csv     # Simulation dataset (1,000 samples)
│
├── models\
│   ├── mlp_system.joblib              # Serialized trained Perceptron model & scalers
│   ├── extended\                      # (Optional) Model trained on all 22 features
│   └── sensor\                        # (Optional) Model trained without coupling factor k
│
├── results\
│   ├── actual_vs_predicted.png        # Scatter plot with y = x reference line
│   ├── training_loss_curves.png       # Training loss convergence history
│   ├── residuals_distribution.png     # Residual error histogram & density curve
│   ├── feature_importance.png         # Permutation feature importance ranking
│   └── evaluation_metrics.json        # Benchmark metrics (R2, RMSE, MAE, MAPE, Max Error)
│
├── train_mlp_system.py                # Main Multiple Perceptron training script (scikit-learn)
├── train_pytorch_system.py            # PyTorch Deep Perceptron implementation
├── predict.py                         # Inference script (CLI params, batch CSV, interactive demo)
├── requirements.txt                   # Python library dependencies
└── README.md                          # Comprehensive documentation & training instructions
```

---

## 5. Prerequisites & Environment Setup

### 1. Requirements
Ensure you have Python 3.10+ installed. The environment requires:
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scipy>=1.10.0`
- `matplotlib>=3.7.0`
- `joblib>=1.3.0`
- *(Optional)* `torch>=2.0.0` (for PyTorch deep modeling)

### 2. Installation
Navigate to the project directory and install dependencies:

```powershell
cd "D:\UROP Research Code"
pip install -r requirements.txt
```

---

## 6. How to Train the ML Model

### Mode 1: Standard Training on Circuit Parameters (Recommended)

Trains the Multiple Perceptron network to predict `Mutual_Inductance_H` based on the 10 circuit parameters:

```powershell
python "D:\UROP Research Code\train_mlp_system.py"
```

**Artifacts Generated:**
- Trained model package: `models/mlp_system.joblib`
- Numerical evaluation metrics: `results/evaluation_metrics.json`
- Plots: `results/actual_vs_predicted.png`, `results/training_loss_curves.png`, `results/residuals_distribution.png`, `results/feature_importance.png`

### Mode 2: Training on All Parameters (22 Features)

Trains the network using all available numeric parameters in the dataset:

```powershell
python "D:\UROP Research Code\train_mlp_system.py" `
  --feature_set all `
  --save_dir "D:\UROP Research Code\models\extended" `
  --results_dir "D:\UROP Research Code\results\extended"
```

### Mode 3: Sensorless / Operating Measurement Mode (Without $k$)

When coupling coefficient $k$ is unknown or unmeasurable in real-time hardware:

```powershell
python "D:\UROP Research Code\train_mlp_system.py" `
  --feature_set sensor `
  --save_dir "D:\UROP Research Code\models\sensor" `
  --results_dir "D:\UROP Research Code\results\sensor"
```

### Mode 4: Custom Feature Selection

You can specify any exact comma-separated subset of features:

```powershell
python "D:\UROP Research Code\train_mlp_system.py" `
  --features "Frequency_kHz,L1_H,L2_H,k,RL_Ohm,Vin_RMS_V"
```

### Advanced Hyperparameter Customization

You can freely tune neural network hyperparameters via command-line arguments:

```powershell
python "D:\UROP Research Code\train_mlp_system.py" `
  --hidden_layers "128,64,32" `
  --activation relu `
  --solver adam `
  --learning_rate_init 0.001 `
  --alpha 0.0001 `
  --max_iter 2000 `
  --test_size 0.2 `
  --random_state 42
```

**Available Hyperparameter Arguments:**
- `--hidden_layers`: Comma-separated neuron counts for hidden layers (default: `"128,64,32"`).
- `--activation`: Non-linear activation function (`relu`, `tanh`, `logistic`).
- `--solver`: Weight optimization algorithm (`adam`, `lbfgs`, `sgd`).
- `--alpha`: L2 weight decay regularization penalty (default: `0.0001`).
- `--learning_rate_init`: Initial learning rate step (default: `0.001`).
- `--max_iter`: Maximum training iterations before early stopping (default: `2000`).
- `--test_size`: Split ratio for unseen holdout validation (default: `0.2` = 200 samples).
- `--random_state`: Random seed for reproducibility (default: `42`).

### Mode 5: PyTorch Deep Learning Training

For deep learning experiments using PyTorch (requires `pip install torch`):

```powershell
python "D:\UROP Research Code\train_pytorch_system.py" --architecture mlp --epochs 300 --batch_size 32
```

---

## 7. Model Performance & Evaluation Benchmarks

The trained Multiple Perceptron System is evaluated on an **unseen holdout test set** (20% test partition, 200 simulation samples).

### Quantitative Metrics Summary

| Feature Configuration | Input Features ($n$) | $R^2$ Score | RMSE ($\text{H}$) | MAE ($\text{H}$) | MAPE (%) | Max Error ($\text{H}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Circuit Parameters (Default)** | **10** | **0.9959** | $1.585 \times 10^{-7}$ | $1.283 \times 10^{-7}$ | **0.70 %** | $5.599 \times 10^{-7}$ |
| **All Features (Extended)** | **22** | **0.9966** | $1.434 \times 10^{-7}$ | $1.050 \times 10^{-7}$ | **0.57 %** | $6.106 \times 10^{-7}$ |
| **Sensor Mode (Without $k$)** | **15** | **0.9901** | $2.453 \times 10^{-7}$ | $1.964 \times 10^{-7}$ | **1.06 %** | $6.362 \times 10^{-7}$ |

### Key Observations:
- **Sub-1% Relative Error**: The default model achieves a Mean Absolute Percentage Error (MAPE) of just **0.70%**, with an $R^2$ of **0.9959**.
- **Accuracy in Microhenries ($\mu\text{H}$)**: The Mean Absolute Error (MAE) is approximately $0.128\,\mu\text{H}$ across typical mutual inductance values ranging between $12.9\,\mu\text{H}$ and $25.1\,\mu\text{H}$.
- **Coupling Estimation without Direct Sensing**: Even when magnetic coupling coefficient $k$ is completely excluded, the network successfully reconstructs $M$ with **99.01% accuracy** ($R^2 = 0.9901$) by leveraging operating voltages, currents, and powers.
- **Feature Importance**: Permutation importance confirms that coupling coefficient $k$, primary coil inductance $L_1$, and secondary coil inductance $L_2$ are the dominant predictors, matching analytical electromagnetic theory ($M = k\sqrt{L_1 L_2}$).

---

## 8. Visualizations Generated

Every training execution automatically exports publication-quality (300 DPI) plots to the `results/` folder:

1. **`actual_vs_predicted.png`**: Scatter plot comparing simulated ground truth mutual inductance against Multiple Perceptron predictions along the ideal $y = x$ reference line.
2. **`training_loss_curves.png`**: Mean squared error loss convergence across iterations demonstrating smooth gradient descent without overfitting.
3. **`residuals_distribution.png`**: Histogram and probability density of error residuals demonstrating a tight zero-mean Gaussian distribution.
4. **`feature_importance.png`**: Permutation feature importance ranking identifying the physical parameters that most strongly govern mutual inductance estimation.

---

## 9. How to Run Inference & Predictions

The `predict.py` script provides three flexible ways to run predictions using the trained model.

### Option A: Interactive Demo Mode
Runs inference on a representative nominal test circuit from the dataset:

```powershell
python "D:\UROP Research Code\predict.py"
```

*Sample Console Output:*
```
[INFO] Loading model package from: D:\UROP Research Code\models\mlp_system.joblib
[INFO] Running Interactive Demo using a representative WPT test circuit...

====================================================================
DEMO TEST CIRCUIT: INPUT PARAMETERS
====================================================================
  Frequency_kHz                : 83.8868
  L1_H                         : 6.814047e-05
  L2_H                         : 8.486213e-05
  k                            : 0.2357
  R1_Ohm                       : 0.0512
  R2_Ohm                       : 0.0617
  RL_Ohm                       : 19.9323
  C1_F                         : 4.491146e-08
  C2_F                         : 4.241694e-08
  Vin_RMS_V                    : 200.4042
--------------------------------------------------------------------
MUTUAL INDUCTANCE PREDICTION RESULTS
--------------------------------------------------------------------
  Predicted Mutual_Inductance_H  : 1.791950e-05 H (17.9195 uH)
  Simulated Ground Truth M    : 1.792147e-05 H (17.9215 uH)
  Absolute Prediction Error   : 1.973730e-09 H (0.0020 uH)
  Relative Percentage Error   : 0.0110 %
  Model Prediction Accuracy   : 99.9890 %
====================================================================
```

### Option B: Command-Line Single Circuit Evaluation
Pass individual circuit component parameters directly through the CLI:

```powershell
python "D:\UROP Research Code\predict.py" `
  --Frequency_kHz 85.0 `
  --L1_H 7.0e-5 `
  --L2_H 8.5e-5 `
  --k 0.25 `
  --R1_Ohm 0.05 `
  --R2_Ohm 0.06 `
  --RL_Ohm 25.0 `
  --C1_F 4.5e-8 `
  --C2_F 4.0e-8 `
  --Vin_RMS_V 200.0
```

*Output:*
```
=================================================================
PREDICTED WPT MUTUAL INDUCTANCE (Multiple Perceptron ML System)
=================================================================
  Mutual_Inductance_H            : 1.922839e-05 H (19.2284 uH)
-----------------------------------------------------------------
  Theoretical M (k * sqrt(L1*L2)) : 1.928406e-05 H (19.2841 uH)
  Absolute Error                  : 5.567362e-08 H (0.0557 uH)
  Relative Percentage Error       : 0.2887 %
=================================================================
```

### Option C: Batch Prediction on a CSV File
To predict mutual inductance for hundreds or thousands of candidate circuit designs:

```powershell
python "D:\UROP Research Code\predict.py" `
  --input_csv "D:\UROP Research Code\data\WPT_1000_Final_Dataset.csv" `
  --output_csv "D:\UROP Research Code\results\predictions.csv"
```

The script automatically appends:
- `Predicted_Mutual_Inductance_H` (in Henries)
- `Predicted_Mutual_Inductance_H_uH` (in microhenries)
- `Theoretical_Mutual_Inductance_H` (if $k, L_1, L_2$ are provided)
- `Prediction_Error_percent`

### Option D: Python API Integration
You can import the prediction engine directly into your own optimization or analysis scripts:

```python
import joblib
import pandas as pd
from predict import predict

# 1. Load trained model package
package = joblib.load(r"D:\UROP Research Code\models\mlp_system.joblib")

# 2. Define input circuit parameters
new_circuit = pd.DataFrame([{
    'Frequency_kHz': 85.0,
    'L1_H': 7.0e-05,
    'L2_H': 8.5e-05,
    'k': 0.25,
    'R1_Ohm': 0.05,
    'R2_Ohm': 0.06,
    'RL_Ohm': 25.0,
    'C1_F': 4.5e-08,
    'C2_F': 4.0e-08,
    'Vin_RMS_V': 200.0
}])

# 3. Predict Mutual Inductance
pred_df = predict(package, new_circuit)
pred_M_Henries = pred_df['Mutual_Inductance_H'].iloc[0]
pred_M_microHenries = pred_M_Henries * 1e6

print(f"Predicted Mutual Inductance: {pred_M_Henries:.6e} H ({pred_M_microHenries:.4f} uH)")
```

---

## 10. Research Applications & Next Steps

1. **Dynamic Air-Gap & Misalignment Estimation**:
   - Because $M$ varies inversely with transmission distance and lateral coil misalignment, this ML surrogate model can estimate real-time spatial positioning in electric vehicle (EV) dynamic wireless charging without physical distance sensors.
2. **Real-Time Closed-Loop Adaptive Tuning**:
   - The neural network inference latency is under $0.1\,\text{ms}$, making it fast enough to run on digital signal processors (DSPs) or microcontrollers for online impedance matching and frequency tracking.
3. **Sensorless Mutual Inductance Reconstruction**:
   - Using the `--feature_set sensor` mode, mutual inductance can be estimated during live operation solely from measured input/output voltages and currents, eliminating the need to measure coupling factor $k$ directly.
