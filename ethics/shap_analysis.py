"""
SHAP Feature Importance Analysis — AI-NIDS Ethics Module
=========================================================
Usage:
  python ethics/shap_analysis.py --model-dir models --output ethics/
"""

import argparse, pickle, os
import numpy as np
import matplotlib.pyplot as plt


def load_models(model_dir):
    with open(f"{model_dir}/ensemble_rf_svm.pkl", "rb") as f:
        ensemble = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return ensemble, scaler


def run_shap_analysis(ensemble, scaler, X_test, feature_names, output_dir):
    rf = ensemble.estimators_[0]  # RandomForest sub-estimator

    try:
        import shap
        print("Running SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(rf)
        sample    = X_test[:min(500, len(X_test))]
        shap_vals = explainer.shap_values(sample)

        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
        else:
            mean_abs = np.abs(shap_vals).mean(axis=0)

    except ImportError:
        print("SHAP not installed — using RF feature importances as proxy")
        mean_abs = rf.feature_importances_

    # Top-15 features
    top_idx   = np.argsort(mean_abs)[::-1][:15]
    top_vals  = mean_abs[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.3, 15))
    ax.barh(top_names[::-1], top_vals[::-1], color=colors)
    ax.set_xlabel("Mean |SHAP| value", fontsize=11)
    ax.set_title("AI-NIDS — SHAP Feature Importance (Top 15)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "shap_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # ── Text report ───────────────────────────────────────────────────────────
    print("\nTop-10 SHAP Features:")
    print(f"{'Rank':>4}  {'Feature':30}  {'Mean |SHAP|':>12}")
    print("-" * 52)
    for i, (name, val) in enumerate(zip(top_names[:10], top_vals[:10]), 1):
        print(f"{i:>4}  {name:30}  {val:12.4f}")

    return dict(zip(top_names, top_vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output",    default="ethics")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ensemble, scaler = load_models(args.model_dir)

    # Generate synthetic test set
    rng = np.random.RandomState(42)
    X_raw = rng.rand(1000, 41)
    X_test = scaler.transform(X_raw)

    feature_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    ]

    run_shap_analysis(ensemble, scaler, X_test, feature_names, args.output)
    print("\n✅ SHAP analysis complete.")


if __name__ == "__main__":
    main()
