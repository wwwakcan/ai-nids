"""
AI-NIDS Demo Alert — SIEM pipeline demonstration
================================================
Usage:
  python src/demo_alert.py --replay data/sample_flows.csv
  python src/demo_alert.py --n 20 --sleep 0.5
"""

import argparse, pickle, time
import numpy as np
import pandas as pd


CLASS_NAMES = {0:'Normal',1:'DoS',2:'Probe',3:'R2L',4:'U2R'}
SEVERITY    = {'U2R':'CRITICAL','R2L':'CRITICAL','DoS':'HIGH','Probe':'MEDIUM','Normal':'INFO'}
COLORS      = {'CRITICAL':'\033[91m','HIGH':'\033[93m','MEDIUM':'\033[94m','INFO':'\033[92m'}
RESET       = '\033[0m'


def load_models(model_dir='models'):
    with open(f"{model_dir}/ensemble_rf_svm.pkl", "rb") as f:
        ensemble = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return ensemble, scaler


def process_flow(features, ensemble, scaler):
    X = scaler.transform(features.reshape(1, -1))
    label_idx = int(ensemble.predict(X)[0])
    label     = CLASS_NAMES[label_idx]
    confidence = float(ensemble.predict_proba(X)[0].max())
    severity   = SEVERITY.get(label, 'MEDIUM')
    return label, confidence, severity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay',    type=str, default=None)
    parser.add_argument('--n',         type=int, default=10)
    parser.add_argument('--sleep',     type=float, default=0.3)
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()

    print("=" * 60)
    print("  AI-NIDS  |  Live Alert Pipeline Demo")
    print("=" * 60)
    print()

    try:
        ensemble, scaler = load_models(args.model_dir)
    except FileNotFoundError:
        print("Models not found — run src/train.py first")
        return

    # Load or generate flows
    if args.replay:
        try:
            df = pd.read_csv(args.replay)
            X_all = df.drop(columns=['label'], errors='ignore').values.astype(float)
            print(f"Replaying {len(X_all)} flows from {args.replay}\n")
        except Exception as e:
            print(f"Could not load {args.replay}: {e}")
            return
    else:
        rng   = np.random.RandomState(42)
        X_all = rng.rand(args.n, 41)
        print(f"Processing {args.n} synthetic flows\n")

    # Process
    counts = {s: 0 for s in ['CRITICAL','HIGH','MEDIUM','INFO']}
    for i, feat in enumerate(X_all[:args.n]):
        t0 = time.time()
        label, conf, severity = process_flow(feat, ensemble, scaler)
        latency = (time.time() - t0) * 1000
        counts[severity] += 1

        color = COLORS.get(severity, '')
        print(f"  Flow {i+1:03d}  |  {color}{severity:8}{RESET}  |  "
              f"{label:8}  |  conf={conf:.3f}  |  {latency:.1f}ms")

        time.sleep(args.sleep)

    print()
    print("─" * 60)
    print("  Summary:")
    for sev, cnt in counts.items():
        color = COLORS.get(sev, '')
        print(f"    {color}{sev:8}{RESET} : {cnt}")
    print("─" * 60)
    print(f"  Total : {sum(counts.values())} flows processed")
    print()
    print("✅ Demo complete. In production, events are forwarded to Elasticsearch.")


if __name__ == "__main__":
    main()
