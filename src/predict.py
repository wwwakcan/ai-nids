"""
AI-NIDS — Single Flow Inference
================================
Usage:
  python src/predict.py --features 0.1 0.9 0.3 ... (41 values)
  python src/predict.py --csv data/sample_flows.csv --row 0
"""

import argparse, pickle
import numpy as np


CLASS_NAMES = {0:'Normal',1:'DoS',2:'Probe',3:'R2L',4:'U2R'}
SEVERITY    = {'U2R':'CRITICAL','R2L':'CRITICAL','DoS':'HIGH','Probe':'MEDIUM','Normal':'INFO'}


def predict_single(features: np.ndarray, model_dir: str = 'models'):
    with open(f"{model_dir}/ensemble_rf_svm.pkl", "rb") as f:
        ensemble = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    X = scaler.transform(features.reshape(1, -1))
    label_idx  = int(ensemble.predict(X)[0])
    label      = CLASS_NAMES[label_idx]
    confidence = float(ensemble.predict_proba(X)[0].max())
    severity   = SEVERITY.get(label, 'MEDIUM')

    # Anomaly score (Isolation Forest)
    ae_score = 0.0
    try:
        with open(f"{model_dir}/isolation_forest.pkl", "rb") as f:
            iforest = pickle.load(f)
        ae_score = float(-iforest.decision_function(X)[0])
    except Exception:
        pass

    return {
        'label':      label,
        'severity':   severity,
        'confidence': round(confidence, 4),
        'ae_score':   round(ae_score, 6),
    }


def main():
    parser = argparse.ArgumentParser(description="AI-NIDS Single Flow Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--features', nargs='+', type=float,
                       metavar='F', help='41 feature values (space-separated)')
    group.add_argument('--csv', type=str, metavar='FILE',
                       help='Path to CSV file containing flow features')
    parser.add_argument('--row',       type=int, default=0, help='Row index in CSV')
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()

    if args.features:
        if len(args.features) < 10:
            parser.error("Provide at least 10 features")
        features = np.array(args.features, dtype=float)
        if len(features) < 41:
            features = np.pad(features, (0, 41 - len(features)))
        features = features[:41]
    else:
        import pandas as pd
        df = pd.read_csv(args.csv)
        feat_cols = [c for c in df.columns if c != 'label']
        features = df[feat_cols].iloc[args.row].values.astype(float)

    result = predict_single(features, args.model_dir)

    print()
    print("=" * 45)
    print("  AI-NIDS Prediction Result")
    print("=" * 45)
    print(f"  Label      : {result['label']}")
    print(f"  Severity   : {result['severity']}")
    print(f"  Confidence : {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
    print(f"  AE Score   : {result['ae_score']:.6f}")
    print("=" * 45)
    print()


if __name__ == "__main__":
    main()
