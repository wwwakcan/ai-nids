"""
AI-NIDS Training Script
========================
Usage:
  python src/train.py --dataset nslkdd --modules all
  python src/train.py --dataset synthetic --modules m1 m2
"""

import argparse, os, pickle, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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

DOS    = ['back','land','neptune','pod','smurf','teardrop','apache2','udpstorm','processtable','worm']
PROBE  = ['ipsweep','nmap','portsweep','satan','mscan','saint']
R2L    = ['ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient',
          'warezmaster','sendmail','named','snmpgetattack','snmpguess','xlock','xsnoop','httptunnel']
U2R    = ['buffer_overflow','loadmodule','perl','rootkit','ps','sqlattack','xterm']
CLASS_MAP = {'Normal':0,'DoS':1,'Probe':2,'R2L':3,'U2R':4}


def map_label(lbl):
    if lbl == 'normal': return 'Normal'
    if lbl in DOS:   return 'DoS'
    if lbl in PROBE: return 'Probe'
    if lbl in R2L:   return 'R2L'
    if lbl in U2R:   return 'U2R'
    return 'DoS'


def load_nslkdd():
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv('data/KDDTrain+.txt', names=COLUMNS, header=None)
    df['label_cat'] = df['label'].apply(map_label)
    for col in ['protocol_type','service','flag']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    feat_cols = [c for c in COLUMNS if c not in ('label','difficulty')]
    X = df[feat_cols].values.astype(float)
    y = df['label_cat'].map(CLASS_MAP).values
    return X, y


def generate_synthetic(n=20000):
    rng = np.random.RandomState(RANDOM_SEED)
    classes = ['Normal','DoS','Probe','R2L','U2R']
    weights = [0.53, 0.25, 0.12, 0.08, 0.02]
    labels  = rng.choice(classes, size=n, p=weights)
    X = np.zeros((n, 41))
    for i, lbl in enumerate(labels):
        if lbl == 'Normal':   X[i] = rng.normal([0.1]*41, [0.05]*41)
        elif lbl == 'DoS':    X[i] = rng.normal([0.9,0.8]+[0.3]*39, [0.05]*41)
        elif lbl == 'Probe':  X[i] = rng.normal([0.2]+[0.6]*40, [0.08]*41)
        elif lbl == 'R2L':    X[i] = rng.normal([0.4]+[0.2]*40, [0.06]*41)
        else:                 X[i] = rng.normal([0.3]*20+[0.7]*21, [0.07]*41)
    return np.clip(X, 0, 1), np.array([CLASS_MAP[l] for l in labels])


def train_m1(X_train, y_train, model_dir):
    print("\n[M1] Training RF + SVM Ensemble...")
    t0 = time.time()
    rf  = RandomForestClassifier(n_estimators=100, max_depth=20,
                                  class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED)
    svc = SVC(C=10, gamma='scale', probability=True, class_weight='balanced', random_state=RANDOM_SEED)
    ens = VotingClassifier([('rf', rf), ('svc', svc)], voting='soft')
    ens.fit(X_train, y_train)
    with open(f"{model_dir}/ensemble_rf_svm.pkl", "wb") as f:
        pickle.dump(ens, f)
    print(f"    Saved ensemble_rf_svm.pkl  [{time.time()-t0:.1f}s]")
    return ens


def train_m2(X_normal, model_dir):
    print("\n[M2] Training Isolation Forest...")
    t0 = time.time()
    iforest = IsolationForest(contamination=0.05, n_estimators=100,
                               random_state=RANDOM_SEED, n_jobs=-1)
    iforest.fit(X_normal)
    with open(f"{model_dir}/isolation_forest.pkl", "wb") as f:
        pickle.dump(iforest, f)
    print(f"    Saved isolation_forest.pkl  [{time.time()-t0:.1f}s]")

    print("[M2] Training LSTM Autoencoder...")
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, Input
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(RANDOM_SEED)
        n_feat  = X_normal.shape[1]
        X3d     = X_normal.reshape(-1, 1, n_feat)
        inp     = Input(shape=(1, n_feat))
        x       = layers.LSTM(64, return_sequences=True)(inp)
        enc     = layers.LSTM(32)(x)
        rep     = layers.RepeatVector(1)(enc)
        x       = layers.LSTM(32, return_sequences=True)(rep)
        x       = layers.LSTM(64, return_sequences=True)(x)
        out     = layers.TimeDistributed(layers.Dense(n_feat, activation='sigmoid'))(x)
        ae      = Model(inp, out)
        ae.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ae.fit(X3d, X3d, epochs=30, batch_size=256, validation_split=0.1, callbacks=[es], verbose=1)
        ae.save(f"{model_dir}/lstm_autoencoder.keras")
        print(f"    Saved lstm_autoencoder.keras")
    except ImportError:
        print("    TensorFlow not installed — skipping LSTM Autoencoder")


def main():
    parser = argparse.ArgumentParser(description="AI-NIDS Training Script")
    parser.add_argument('--dataset', choices=['nslkdd','synthetic'], default='nslkdd')
    parser.add_argument('--modules', nargs='+', choices=['m1','m2','all'], default=['all'])
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Dataset : {args.dataset}")
    print(f"Modules : {args.modules}")

    # Load data
    if args.dataset == 'nslkdd':
        print("\nLoading NSL-KDD...")
        X_raw, y = load_nslkdd()
    else:
        print("\nGenerating synthetic data...")
        X_raw, y = generate_synthetic()

    # Scale
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    with open(f"{args.model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved scaler.pkl")

    # SMOTE
    X_res, y_res = SMOTE(random_state=RANDOM_SEED, k_neighbors=3).fit_resample(X, y)
    X_normal = X_res[y_res == 0]

    modules = ['m1','m2'] if 'all' in args.modules else args.modules

    if 'm1' in modules:
        train_m1(X_res, y_res, args.model_dir)
    if 'm2' in modules:
        train_m2(X_normal, args.model_dir)

    print("\n✅ Training complete. Models saved to:", args.model_dir)


if __name__ == "__main__":
    main()
