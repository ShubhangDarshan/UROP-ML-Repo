"""
Inference & Prediction Script for WPT Mutual Inductance ML System
================================================================
Loads the saved Multiple Perceptron model and predicts Mutual_Inductance_H based on:
  1. Individual circuit parameters passed via CLI arguments.
  2. Batch CSV files containing multiple circuit parameter configurations.
  3. Interactive Demo mode using a verified sample from the WPT dataset.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "mlp_system.joblib")
DEFAULT_OUTPUT_CSV = os.path.join(SCRIPT_DIR, "results", "predictions.csv")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference to predict Mutual Inductance (Mutual_Inductance_H) using trained ML model."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved trained model package (.joblib)."
    )
    parser.add_argument(
        '--input_csv',
        type=str,
        default=None,
        help="Path to an input CSV file containing circuit parameters for batch prediction."
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to save batch predictions CSV."
    )

    # Physical circuit input parameter arguments
    parser.add_argument('--Frequency_kHz', type=float, default=None, help="Operating Frequency (kHz)")
    parser.add_argument('--L1_H', type=float, default=None, help="Primary Inductance L1 (H)")
    parser.add_argument('--L2_H', type=float, default=None, help="Secondary Inductance L2 (H)")
    parser.add_argument('--k', type=float, default=None, help="Magnetic Coupling Coefficient k (0.0 to 1.0)")
    parser.add_argument('--R1_Ohm', type=float, default=None, help="Primary Coil AC Resistance R1 (Ohm)")
    parser.add_argument('--R2_Ohm', type=float, default=None, help="Secondary Coil AC Resistance R2 (Ohm)")
    parser.add_argument('--RL_Ohm', type=float, default=None, help="Load Resistance RL (Ohm)")
    parser.add_argument('--C1_F', type=float, default=None, help="Primary Capacitance C1 (F)")
    parser.add_argument('--C2_F', type=float, default=None, help="Secondary Capacitance C2 (F)")
    parser.add_argument('--Vin_RMS_V', type=float, default=None, help="Inverter Input RMS Voltage (V)")

    # Optional electrical operating parameters (for models trained on extended/sensor features)
    parser.add_argument('--Iin_RMS_A', type=float, default=None, help="RMS Input Current (A)")
    parser.add_argument('--Vout_RMS_V', type=float, default=None, help="RMS Output Voltage (V)")
    parser.add_argument('--Iout_RMS_A', type=float, default=None, help="RMS Output Current (A)")
    parser.add_argument('--Pin_W', type=float, default=None, help="Active Input Power (W)")
    parser.add_argument('--Pout_W', type=float, default=None, help="Active Output Power (W)")
    parser.add_argument('--Efficiency_percent', type=float, default=None, help="Power Transfer Efficiency (percent)")
    parser.add_argument('--Power_Factor', type=float, default=None, help="System Power Factor")
    parser.add_argument('--Apparent_Power_VA', type=float, default=None, help="Apparent Power (VA)")

    return parser.parse_args()


def load_package(model_path):
    """Loads the serialized ML model package from disk."""
    if not os.path.exists(model_path):
        fallback = os.path.join(SCRIPT_DIR, "models", "mlp_system.joblib")
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Please train the model first by running:\n"
                f"    python train_mlp_system.py"
            )
    print(f"[INFO] Loading model package from: {model_path}")
    return joblib.load(model_path)


def predict(package, input_df):
    """
    Executes inference on input_df based on the package's feature set and architecture.
    Returns a DataFrame containing the predicted values.
    """
    features = package['features']
    mode = package.get('mode', 'single_target')
    models = package['models']
    scaler_X = package['scaler_X']
    scalers_y = package['scalers_y']
    targets = package['targets']

    missing = [c for c in features if c not in input_df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required feature columns expected by the model:\n"
            f"Missing: {missing}\nRequired: {features}"
        )

    X = input_df[features].copy()
    X_scaled = scaler_X.transform(X)

    if mode == 'single_target':
        target_name = targets[0]
        mlp = models[target_name]
        scaler_y = scalers_y[target_name]
        p_scaled = mlp.predict(X_scaled).reshape(-1, 1)
        p = scaler_y.inverse_transform(p_scaled).ravel()
        pred_df = pd.DataFrame({target_name: p}, index=input_df.index)

    elif mode == 'multi_target':
        mlp = models['MultiTarget_MLP']
        scaler_y = scalers_y['unified']
        y_pred_scaled = mlp.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        pred_df = pd.DataFrame(y_pred, columns=targets, index=input_df.index)

    elif mode == 'dedicated_perceptrons':
        pred_cols = {}
        for target_name in targets:
            mlp = models[target_name]
            scaler_y = scalers_y[target_name]
            p_scaled = mlp.predict(X_scaled).reshape(-1, 1)
            p = scaler_y.inverse_transform(p_scaled).ravel()
            pred_cols[target_name] = p
        pred_df = pd.DataFrame(pred_cols, index=input_df.index)

    else:
        raise ValueError(f"Unknown architecture mode in package: {mode}")

    return pred_df


def main():
    args = parse_arguments()
    package = load_package(args.model_path)
    required_features = package['features']
    targets = package['targets']

    # ============================================================
    # Case 1: Batch Prediction from CSV
    # ============================================================
    if args.input_csv is not None:
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"Input CSV file '{args.input_csv}' does not exist.")
        print(f"[INFO] Reading batch data from: {args.input_csv}")
        input_df = pd.read_csv(args.input_csv)

        pred_df = predict(package, input_df)

        result_df = input_df.copy()
        for col in pred_df.columns:
            result_df[f"Predicted_{col}"] = pred_df[col]
            if "Inductance" in col or "_H" in col:
                result_df[f"Predicted_{col}_uH"] = pred_df[col] * 1e6

        # If coupling k and inductances L1, L2 are present, compute theoretical reference
        if all(c in input_df.columns for c in ['k', 'L1_H', 'L2_H']):
            theo_M = input_df['k'] * np.sqrt(input_df['L1_H'] * input_df['L2_H'])
            result_df["Theoretical_Mutual_Inductance_H"] = theo_M
            result_df["Theoretical_Mutual_Inductance_uH"] = theo_M * 1e6
            if 'Mutual_Inductance_H' in pred_df.columns:
                err_pct = np.abs(pred_df['Mutual_Inductance_H'] - theo_M) / theo_M * 100.0
                result_df["Prediction_Error_percent"] = err_pct

        out_dir = os.path.dirname(args.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        result_df.to_csv(args.output_csv, index=False)
        print(f"[INFO] Saved batch predictions to: {args.output_csv}")
        print("\nFirst 5 Predicted Rows:")
        display_cols = [c for c in result_df.columns if "Predicted" in c or c in ['Mutual_Inductance_H', 'k', 'L1_H', 'L2_H']]
        print(result_df[display_cols].head())
        return

    # ============================================================
    # Case 2: Individual Parameters Passed via CLI
    # ============================================================
    cli_dict = {
        'Frequency_kHz': args.Frequency_kHz,
        'L1_H': args.L1_H,
        'L2_H': args.L2_H,
        'k': args.k,
        'R1_Ohm': args.R1_Ohm,
        'R2_Ohm': args.R2_Ohm,
        'RL_Ohm': args.RL_Ohm,
        'C1_F': args.C1_F,
        'C2_F': args.C2_F,
        'Vin_RMS_V': args.Vin_RMS_V,
        'Iin_RMS_A': args.Iin_RMS_A,
        'Vout_RMS_V': args.Vout_RMS_V,
        'Iout_RMS_A': args.Iout_RMS_A,
        'Pin_W': args.Pin_W,
        'Pout_W': args.Pout_W,
        'Efficiency_percent': args.Efficiency_percent,
        'Power_Factor': args.Power_Factor,
        'Apparent_Power_VA': args.Apparent_Power_VA
    }

    provided_cli = {k: v for k, v in cli_dict.items() if v is not None}

    if len(provided_cli) > 0:
        missing = [f for f in required_features if f not in provided_cli]
        if missing:
            print(f"[ERROR] Missing required parameters expected by the model:")
            print(f"  Missing  : {missing}")
            print(f"  Required : {required_features}")
            print("\nPlease provide all required CLI flags or run without arguments for Demo mode.")
            return

        sample_df = pd.DataFrame([provided_cli])
        pred_df = predict(package, sample_df)

        print("\n" + "=" * 65)
        print("PREDICTED WPT MUTUAL INDUCTANCE (Multiple Perceptron ML System)")
        print("=" * 65)
        for col in pred_df.columns:
            val = pred_df[col].iloc[0]
            val_uH = val * 1e6 if ("Inductance" in col or "_H" in col) else None
            if val_uH is not None:
                print(f"  {col:<30} : {val:.6e} H ({val_uH:.4f} uH)")
            else:
                print(f"  {col:<30} : {val:.4f}")

        # If theoretical formula can be evaluated
        if all(k in provided_cli for k in ['k', 'L1_H', 'L2_H']) and 'Mutual_Inductance_H' in pred_df.columns:
            theo_M = provided_cli['k'] * np.sqrt(provided_cli['L1_H'] * provided_cli['L2_H'])
            pred_M = pred_df['Mutual_Inductance_H'].iloc[0]
            abs_err = abs(pred_M - theo_M)
            rel_err_pct = (abs_err / theo_M) * 100.0

            print("-" * 65)
            print(f"  Theoretical M (k * sqrt(L1*L2)) : {theo_M:.6e} H ({theo_M * 1e6:.4f} uH)")
            print(f"  Absolute Error                  : {abs_err:.6e} H ({abs_err * 1e6:.4f} uH)")
            print(f"  Relative Percentage Error       : {rel_err_pct:.4f} %")
        print("=" * 65 + "\n")
        return

    # ============================================================
    # Case 3: Interactive Demo Mode
    # ============================================================
    print("\n[INFO] No CLI parameters or input CSV specified.")
    print("[INFO] Running Interactive Demo using a representative WPT test circuit...")

    dataset_path = os.path.join(SCRIPT_DIR, "data", "WPT_1000_Final_Dataset.csv")
    if os.path.exists(dataset_path):
        df_all = pd.read_csv(dataset_path)
        if 'Simulation_Status' in df_all.columns:
            df_all = df_all[df_all['Simulation_Status'] == 'OK']
        sample_row = df_all.iloc[0]
        demo_dict = {f: sample_row[f] for f in required_features if f in sample_row}
        ground_truth_M = sample_row.get('Mutual_Inductance_H', None)
    else:
        demo_dict = {
            'Frequency_kHz': 83.8868,
            'L1_H': 6.8140e-05,
            'L2_H': 8.4862e-05,
            'k': 0.235675,
            'R1_Ohm': 0.0512,
            'R2_Ohm': 0.0617,
            'RL_Ohm': 19.9323,
            'C1_F': 4.4911e-08,
            'C2_F': 4.2417e-08,
            'Vin_RMS_V': 200.4042
        }
        ground_truth_M = 1.792147e-05

    sample_df = pd.DataFrame([demo_dict])
    pred_df = predict(package, sample_df)

    print("\n" + "=" * 68)
    print("DEMO TEST CIRCUIT: INPUT PARAMETERS")
    print("=" * 68)
    for k, v in demo_dict.items():
        if abs(v) < 0.001 and v != 0:
            print(f"  {k:<28} : {v:.6e}")
        else:
            print(f"  {k:<28} : {v:.4f}")
    print("-" * 68)
    print("MUTUAL INDUCTANCE PREDICTION RESULTS")
    print("-" * 68)
    for col in pred_df.columns:
        val = pred_df[col].iloc[0]
        val_uH = val * 1e6 if ("Inductance" in col or "_H" in col) else None
        if val_uH is not None:
            print(f"  Predicted {col:<20} : {val:.6e} H ({val_uH:.4f} uH)")
        else:
            print(f"  Predicted {col:<20} : {val:.4f}")

    if ground_truth_M is not None and 'Mutual_Inductance_H' in pred_df.columns:
        pred_M = pred_df['Mutual_Inductance_H'].iloc[0]
        abs_err = abs(pred_M - ground_truth_M)
        rel_err_pct = (abs_err / ground_truth_M) * 100.0
        acc_pct = 100.0 - rel_err_pct

        print(f"  Simulated Ground Truth M    : {ground_truth_M:.6e} H ({ground_truth_M * 1e6:.4f} uH)")
        print(f"  Absolute Prediction Error   : {abs_err:.6e} H ({abs_err * 1e6:.4f} uH)")
        print(f"  Relative Percentage Error   : {rel_err_pct:.4f} %")
        print(f"  Model Prediction Accuracy   : {acc_pct:.4f} %")
    print("=" * 68 + "\n")


if __name__ == '__main__':
    main()
