"""
AI-NIDS Evaluation Script
==========================
Usage:
  python src/evaluate.py --test-split 0.2 --adversarial --epsilon 0.05
"""

import argparse, pickle
import numpy as np
from sklearn.metrics import (classification_report, accuracy_score,
                              f1_score, confusion_matrix)

RANDOM_SEED = 42
CLASS_NAMES = ['Normal','DoS','Probe','R2L','U2R']


def fgsm_tabular(X, importance, epsilon):
    sign_grad = np.sign(importance / (importance.max() + 1e-9))
    return np.clip(X + epsilon * sign_grad, 0, 1)


def feature_squeezing(X, bits=4):
    levels = 2 ** bits - 1
    return np.round(X * levels) / levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-split',  type=float, default=0.2)
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--epsilon',     type=float, default=0.05)
    parser.add_argument('--model-dir',   default='models')
    args = parser.parse_args()

    # Load models
    with open(f"{args.model_dir}/ensemble_rf_svm.pkl", "rb") as f:
        ensemble = pickle.load(f)
    with open(f"{args.model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load test data (synthetic fallback)
    try:
        import pandas as pd
        COLUMNS = [
            'duration','protocol_type','service','flag','src_bytes','dst_bytes',
            'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
            'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login',
            'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
            'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
            'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
            'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
            'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
        ]
        from sklearn.preprocessing import LabelEncoder
        df = pd.read_csv('data/KDDTest+.txt', names=COLUMNS, header=None)
        for col in ['protocol_type','service','flag']:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        feat_cols = [c for c in COLUMNS if c not in ('label','difficulty')]
        X_raw = df[feat_cols].values.astype(float)
        from src.train import map_label, CLASS_MAP
        y = df['label'].apply(map_label).map(CLASS_MAP).values
    except Exception:
        print("Using synthetic test data...")
        from src.train import generate_synthetic
        from sklearn.model_selection import train_test_split
        X_all, y_all = generate_synthetic(5000)
        _, X_raw, _, y = train_test_split(X_all, y_all, test_size=args.test_split,
                                          stratify=y_all, random_state=RANDOM_SEED)

    X_test = scaler.transform(X_raw)

    # ── Standard evaluation ──────────────────────────────────────────────────
    y_pred = ensemble.predict(X_test)
    acc    = accuracy_score(y, y_pred)
    f1     = f1_score(y, y_pred, average='macro', zero_division=0)

    print("=" * 60)
    print("        AI-NIDS — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (macro): {f1:.4f}")
    print()
    print(classification_report(y, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # ── Adversarial evaluation ───────────────────────────────────────────────
    if args.adversarial:
        importance = ensemble.estimators_[0].feature_importances_
        epsilons   = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]

        print("=" * 60)
        print("        ADVERSARIAL ROBUSTNESS BENCHMARK")
        print("=" * 60)
        print(f"{'Epsilon':>10} | {'No Defence':>12} | {'Feat.Squeezing':>15}")
        print("-" * 45)

        for eps in epsilons:
            X_adv    = fgsm_tabular(X_test, importance, eps) if eps > 0 else X_test.copy()
            acc_base = accuracy_score(y, ensemble.predict(X_adv))
            X_sq     = feature_squeezing(X_adv, bits=4)
            acc_fs   = accuracy_score(y, ensemble.predict(X_sq))
            print(f"{eps:>10.2f} | {acc_base:>12.4f} | {acc_fs:>15.4f}")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
