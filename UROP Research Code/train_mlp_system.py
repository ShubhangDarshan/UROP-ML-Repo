"""
Wireless Power Transfer (WPT) - Multiple Perceptron Machine Learning System
===========================================================================
This module implements a Multiple Perceptron Machine Learning System to model
and predict the Mutual Inductance (Mutual_Inductance_H) of Inductive Wireless
Power Transfer systems based on other physical circuit and system parameters.

Supported Modes:
  1. single_target (default): High-precision Multi-Layer Perceptron (MLP)
     predicting Mutual_Inductance_H.
  2. multi_target: Unified MLP network mapping input parameters to multiple targets.
  3. dedicated_perceptrons: System of independent Perceptron networks for multiple targets.

Feature Sets Supported:
  - circuit (default): 10 physical circuit design parameters
    (Frequency_kHz, L1_H, L2_H, k, R1_Ohm, R2_Ohm, RL_Ohm, C1_F, C2_F, Vin_RMS_V).
  - all: All valid numerical parameters available in the dataset.
  - sensor: Operating electrical parameters without coupling factor k (for sensorless estimation).
  - custom: User-specified comma-separated list of features via --features.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    max_error
)
from sklearn.inspection import permutation_importance


# ==========================================
# Paths & Default Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "WPT_1000_Final_Dataset.csv")
DEFAULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "models")
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

CIRCUIT_FEATURES = [
    'Frequency_kHz',       # Inverter operating frequency (kHz)
    'L1_H',                # Primary coil self-inductance (H)
    'L2_H',                # Secondary coil self-inductance (H)
    'k',                   # Magnetic coupling coefficient (0.0 - 1.0)
    'R1_Ohm',              # Primary coil AC resistance (Ohm)
    'R2_Ohm',              # Secondary coil AC resistance (Ohm)
    'RL_Ohm',              # Load resistance (Ohm)
    'C1_F',                # Primary series compensation capacitance (F)
    'C2_F',                # Secondary series compensation capacitance (F)
    'Vin_RMS_V'            # Inverter RMS input voltage (V)
]

SENSOR_FEATURES = [
    'Frequency_kHz', 'L1_H', 'L2_H', 'R1_Ohm', 'R2_Ohm',
    'RL_Ohm', 'C1_F', 'C2_F', 'Vin_RMS_V', 'Iin_RMS_A',
    'Vout_RMS_V', 'Iout_RMS_A', 'Pin_W', 'Pout_W', 'Efficiency_percent'
]

DEFAULT_FEATURES = CIRCUIT_FEATURES

DEFAULT_TARGETS = [
    'Mutual_Inductance_H'  # Mutual Inductance between coils (H)
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Multiple Perceptron System to predict Mutual Inductance (Mutual_Inductance_H) on WPT Dataset."
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the dataset CSV file."
    )
    parser.add_argument(
        '--feature_set',
        type=str,
        choices=['circuit', 'all', 'sensor'],
        default='circuit',
        help="Predefined feature set: 'circuit' (10 circuit parameters), 'all' (all dataset features), or 'sensor' (without k)."
    )
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help="Comma-separated custom feature names to input into the model (overrides --feature_set)."
    )
    parser.add_argument(
        '--target',
        type=str,
        default='Mutual_Inductance_H',
        help="Target column to predict (default: 'Mutual_Inductance_H')."
    )
    parser.add_argument(
        '--targets',
        type=str,
        default=None,
        help="Comma-separated target column names if predicting multiple targets (overrides --target)."
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single_target', 'multi_target', 'dedicated_perceptrons'],
        default='single_target',
        help="Architecture mode: 'single_target', 'multi_target', or 'dedicated_perceptrons'."
    )
    parser.add_argument(
        '--hidden_layers',
        type=str,
        default='128,64,32',
        help="Comma-separated neuron counts for hidden layers (e.g., '128,64,32')."
    )
    parser.add_argument(
        '--activation',
        type=str,
        choices=['relu', 'tanh', 'logistic'],
        default='relu',
        help="Perceptron non-linear activation function."
    )
    parser.add_argument(
        '--solver',
        type=str,
        choices=['adam', 'lbfgs', 'sgd'],
        default='adam',
        help="Optimization algorithm for training weights."
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.0001,
        help="L2 regularization penalty parameter."
    )
    parser.add_argument(
        '--learning_rate_init',
        type=float,
        default=0.001,
        help="Initial learning rate."
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=2000,
        help="Maximum training epochs/iterations."
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing (default: 0.2)."
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help="Random seed for data split and weight initialization."
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Directory to store trained model artifacts."
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to store evaluation plots and metrics."
    )
    return parser.parse_args()


def load_and_preprocess_data(data_path, feature_set='circuit', custom_features=None, targets=['Mutual_Inductance_H']):
    """
    Loads dataset, filters valid simulation rows, resolves input features,
    drops zero-variance columns, and returns X and y.
    """
    if not os.path.exists(data_path):
        fallback = os.path.join(SCRIPT_DIR, "data", "WPT_1000_Final_Dataset.csv")
        if os.path.exists(fallback):
            print(f"[INFO] Path '{data_path}' not found. Falling back to '{fallback}'.")
            data_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at '{data_path}' or fallback.")

    print(f"[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Raw dataset shape: {df.shape}")

    # Filter by Simulation_Status if present
    if 'Simulation_Status' in df.columns:
        valid_mask = df['Simulation_Status'] == 'OK'
        df = df[valid_mask].copy()
        print(f"[INFO] Rows with Simulation_Status == 'OK': {len(df)}")

    # Determine input features
    if custom_features is not None:
        features = [f.strip() for f in custom_features.split(',') if f.strip()]
    elif feature_set == 'circuit':
        features = CIRCUIT_FEATURES.copy()
    elif feature_set == 'sensor':
        features = SENSOR_FEATURES.copy()
    elif feature_set == 'all':
        ignore_cols = set(targets + ['Simulation_Status', 'Rejection_Reason', 'ESR_C1_Ohm', 'ESR_C2_Ohm'])
        features = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    else:
        features = CIRCUIT_FEATURES.copy()

    # Verify requested columns exist
    missing_feats = [col for col in features if col not in df.columns]
    if missing_feats:
        raise ValueError(f"Features missing in dataset: {missing_feats}")

    missing_targets = [col for col in targets if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Targets missing in dataset: {missing_targets}")

    X = df[features].copy()
    y = df[targets].copy()

    # Drop zero-variance columns in X if any
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        print(f"[WARNING] Dropping zero-variance feature columns: {constant_cols}")
        X = X.drop(columns=constant_cols)
        features = [c for c in features if c not in constant_cols]

    print(f"[INFO] Input features ({len(features)}): {features}")
    print(f"[INFO] Target output(s) ({len(targets)}): {targets}")
    return X, y, features, targets


def build_perceptron_regressor(hidden_layers, activation, solver, alpha, lr_init, max_iter, random_state):
    """Constructs a configured Multi-Layer Perceptron Regressor."""
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        verbose=False
    )


def compute_metrics(y_true, y_pred, target_names):
    """Calculates comprehensive regression performance metrics."""
    metrics = {}
    for i, name in enumerate(target_names):
        yt = y_true[:, i] if (isinstance(y_true, np.ndarray) and y_true.ndim > 1) else np.asarray(y_true).ravel()
        yp = y_pred[:, i] if (isinstance(y_pred, np.ndarray) and y_pred.ndim > 1) else np.asarray(y_pred).ravel()

        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)
        mape = mean_absolute_percentage_error(yt, yp) * 100.0
        me = max_error(yt, yp)

        metrics[name] = {
            'R2_Score': float(r2),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE_percent': float(mape),
            'Max_Error': float(me)
        }
    return metrics


def format_val(val):
    """Formats numeric values adaptively in scientific or fixed-point notation."""
    if abs(val) < 0.001 and val != 0:
        return f"{val:.4e}"
    return f"{val:.4f}"


def print_metrics_table(metrics):
    """Prints a formatted ASCII table of evaluation metrics."""
    header = f"{'Target Variable':<24} | {'R2 Score':<10} | {'RMSE':<14} | {'MAE':<14} | {'MAPE (%)':<10} | {'Max Error':<14}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(header)
    print(divider)
    for target, m in metrics.items():
        print(f"{target:<24} | {m['R2_Score']:<10.4f} | {format_val(m['RMSE']):<14} | {format_val(m['MAE']):<14} | {m['MAPE_percent']:<10.2f} | {format_val(m['Max_Error']):<14}")
    print(divider + "\n")


def plot_actual_vs_predicted(y_true, y_pred, target_names, out_path):
    """Generates and saves publication-quality Actual vs. Predicted scatter plots."""
    n_targets = len(target_names)
    cols = min(3, n_targets)
    rows = int(np.ceil(n_targets / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows), squeeze=False)
    fig.suptitle("Multiple Perceptron System: Actual vs. Predicted Performance", fontsize=14, fontweight='bold')

    for idx, name in enumerate(target_names):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        yt = y_true[:, idx] if (isinstance(y_true, np.ndarray) and y_true.ndim > 1) else np.asarray(y_true).ravel()
        yp = y_pred[:, idx] if (isinstance(y_pred, np.ndarray) and y_pred.ndim > 1) else np.asarray(y_pred).ravel()
        r2 = r2_score(yt, yp)

        unit_label = " (H)" if "Inductance" in name or "_H" in name else ""

        ax.scatter(yt, yp, alpha=0.6, color='#1f77b4', edgecolors='k', linewidth=0.5, s=40, label='Test Samples')

        min_val = min(np.min(yt), np.min(yp))
        max_val = max(np.max(yt), np.max(yp))
        margin = 0.05 * (max_val - min_val) if max_val != min_val else 0.01 * max_val
        ref_line = np.linspace(min_val - margin, max_val + margin, 100)
        ax.plot(ref_line, ref_line, 'r--', linewidth=1.5, label='Ideal (y = x)')

        ax.set_title(f"{name}\n($R^2 = {r2:.4f}$)", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"Simulated Ground Truth{unit_label}", fontsize=10)
        ax.set_ylabel(f"Perceptron Prediction{unit_label}", fontsize=10)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', frameon=True)

    for idx in range(n_targets, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r, c])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Actual vs. Predicted plot to: {out_path}")


def plot_loss_curves(models_dict, out_path):
    """Plots training loss curve over iterations."""
    plt.figure(figsize=(8, 5))
    for name, model in models_dict.items():
        if hasattr(model, 'loss_curve_'):
            plt.plot(model.loss_curve_, label=f"{name} ({len(model.loss_curve_)} epochs)", linewidth=1.5)
    plt.title("Multiple Perceptron System: Training Loss Convergence", fontsize=13, fontweight='bold')
    plt.xlabel("Iteration / Epoch", fontsize=11)
    plt.ylabel("Loss (Squared Error)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Loss Curves plot to: {out_path}")


def plot_residuals(y_true, y_pred, target_names, out_path):
    """Plots distribution of prediction residuals (errors)."""
    n_targets = len(target_names)
    cols = min(3, n_targets)
    rows = int(np.ceil(n_targets / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5 * rows), squeeze=False)
    fig.suptitle("Multiple Perceptron System: Residual Error Distributions", fontsize=14, fontweight='bold')

    for idx, name in enumerate(target_names):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        yt = y_true[:, idx] if (isinstance(y_true, np.ndarray) and y_true.ndim > 1) else np.asarray(y_true).ravel()
        yp = y_pred[:, idx] if (isinstance(y_pred, np.ndarray) and y_pred.ndim > 1) else np.asarray(y_pred).ravel()
        residuals = yt - yp

        unit_label = " [H]" if "Inductance" in name or "_H" in name else ""

        ax.hist(residuals, bins=25, color='#2ca02c', alpha=0.7, edgecolor='black', density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
        ax.set_title(f"Residuals: {name}", fontsize=11, fontweight='bold')
        ax.set_xlabel(f"Error (Actual - Predicted){unit_label}", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', frameon=True)

    for idx in range(n_targets, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r, c])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Residuals plot to: {out_path}")


def plot_feature_importance(model, X_test_scaled, y_test_scaled, feature_names, out_path):
    """Computes and plots Permutation Feature Importance for input parameters."""
    print("[INFO] Computing Permutation Feature Importance...")
    result = permutation_importance(model, X_test_scaled, y_test_scaled, n_repeats=5, random_state=42)
    sorted_idx = result.importances_mean.argsort()

    plt.figure(figsize=(9, max(5, len(feature_names) * 0.35)))
    plt.barh(np.array(feature_names)[sorted_idx], result.importances_mean[sorted_idx],
             xerr=result.importances_std[sorted_idx], color='#3b528b', edgecolor='black', alpha=0.85, capsize=4)
    plt.title("Permutation Feature Importance (Impact on Perceptron Accuracy)", fontsize=13, fontweight='bold')
    plt.xlabel("Mean Importance (Drop in Model Performance)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6, axis='x')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Feature Importance plot to: {out_path}")


def main():
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Determine targets based on arguments
    if args.targets is not None:
        targets = [t.strip() for t in args.targets.split(',') if t.strip()]
        mode = args.mode if args.mode != 'single_target' else 'multi_target'
    elif args.target is not None:
        targets = [args.target]
        mode = 'single_target'
    else:
        targets = DEFAULT_TARGETS
        mode = 'single_target'

    # If multiple targets requested but mode is single_target, switch to multi_target
    if len(targets) > 1 and mode == 'single_target':
        mode = 'multi_target'

    # 2. Parse hidden layer architecture
    hidden_layer_sizes = tuple(int(x.strip()) for x in args.hidden_layers.split(','))

    # 3. Load & Preprocess Data
    X_df, y_df, active_features, active_targets = load_and_preprocess_data(
        args.data_path,
        feature_set=args.feature_set,
        custom_features=args.features,
        targets=targets
    )

    print("=" * 75)
    print("WIRELESS POWER TRANSFER: MUTUAL INDUCTANCE ML TRAINING PIPELINE")
    print("=" * 75)
    print(f"Target Variable(s)  : {active_targets}")
    print(f"Architecture Mode   : {mode}")
    print(f"Input Feature Count : {len(active_features)}")
    print(f"Perceptron Topology : Input({len(active_features)}) -> {hidden_layer_sizes} -> Output({len(active_targets)})")
    print(f"Activation Function : {args.activation}")
    print(f"Optimizer (Solver)  : {args.solver} (alpha={args.alpha}, lr={args.learning_rate_init})")
    print("=" * 75)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=args.test_size, random_state=args.random_state
    )
    print(f"[INFO] Train Samples: {len(X_train)}, Test Samples: {len(X_test)}")

    # Feature Normalization
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # 4. Model Training according to Mode
    models_dict = {}
    scalers_y_dict = {}

    if mode == 'single_target':
        target_name = active_targets[0]
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)

        mlp = build_perceptron_regressor(
            hidden_layer_sizes, args.activation, args.solver,
            args.alpha, args.learning_rate_init, args.max_iter, args.random_state
        )
        print(f"\n[INFO] Training Perceptron Network for Target '{target_name}'...")
        mlp.fit(X_train_scaled, y_train_scaled.ravel())
        print(f"[INFO] Finished training after {mlp.n_iter_} iterations.")

        y_pred_scaled = mlp.predict(X_test_scaled).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_eval = y_test.values

        models_dict[target_name] = mlp
        scalers_y_dict[target_name] = scaler_y

        # Feature importance
        plot_feature_importance(
            mlp, X_test_scaled, scaler_y.transform(y_test), active_features,
            os.path.join(args.results_dir, "feature_importance.png")
        )

    elif mode == 'multi_target':
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        mlp = build_perceptron_regressor(
            hidden_layer_sizes, args.activation, args.solver,
            args.alpha, args.learning_rate_init, args.max_iter, args.random_state
        )
        print("\n[INFO] Training Unified Multi-Output Perceptron Network...")
        mlp.fit(X_train_scaled, y_train_scaled)
        print(f"[INFO] Finished training after {mlp.n_iter_} iterations.")

        y_pred_scaled = mlp.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_eval = y_test.values

        models_dict['MultiTarget_MLP'] = mlp
        scalers_y_dict['unified'] = scaler_y

        # Feature importance
        plot_feature_importance(
            mlp, X_test_scaled, y_test_scaled, active_features,
            os.path.join(args.results_dir, "feature_importance.png")
        )

    elif mode == 'dedicated_perceptrons':
        y_pred_list = []
        y_test_eval = y_test.values

        for idx, target_name in enumerate(active_targets):
            print(f"\n[INFO] Training Dedicated Perceptron for '{target_name}'...")
            scaler_y = StandardScaler()
            y_tr = scaler_y.fit_transform(y_train[[target_name]]).ravel()

            mlp = build_perceptron_regressor(
                hidden_layer_sizes, args.activation, args.solver,
                args.alpha, args.learning_rate_init, args.max_iter, args.random_state + idx
            )
            mlp.fit(X_train_scaled, y_tr)
            print(f"[INFO] '{target_name}' trained in {mlp.n_iter_} iterations.")

            pred_scaled = mlp.predict(X_test_scaled).reshape(-1, 1)
            pred = scaler_y.inverse_transform(pred_scaled).ravel()
            y_pred_list.append(pred)

            models_dict[target_name] = mlp
            scalers_y_dict[target_name] = scaler_y

        y_pred = np.column_stack(y_pred_list)

    # 5. Evaluate Performance
    metrics = compute_metrics(y_test_eval, y_pred, active_targets)
    print_metrics_table(metrics)

    # Save metrics JSON
    metrics_path = os.path.join(args.results_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Saved evaluation metrics to: {metrics_path}")

    # 6. Generate Evaluation Plots
    plot_actual_vs_predicted(
        y_test_eval, y_pred, active_targets,
        os.path.join(args.results_dir, "actual_vs_predicted.png")
    )
    plot_loss_curves(
        models_dict,
        os.path.join(args.results_dir, "training_loss_curves.png")
    )
    plot_residuals(
        y_test_eval, y_pred, active_targets,
        os.path.join(args.results_dir, "residuals_distribution.png")
    )

    # 7. Save Model Artifacts
    package = {
        'mode': mode,
        'feature_set': args.feature_set,
        'models': models_dict,
        'scaler_X': scaler_X,
        'scalers_y': scalers_y_dict,
        'features': active_features,
        'targets': active_targets,
        'hyperparameters': {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': args.activation,
            'solver': args.solver,
            'alpha': args.alpha,
            'learning_rate_init': args.learning_rate_init,
            'max_iter': args.max_iter
        },
        'metrics': metrics
    }
    model_save_path = os.path.join(args.save_dir, "mlp_system.joblib")
    joblib.dump(package, model_save_path)
    print(f"[INFO] Successfully exported trained ML Model package to: {model_save_path}")
    print("=" * 75)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("=" * 75)


if __name__ == '__main__':
    main()
