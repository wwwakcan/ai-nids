"""
Unit tests for AI-NIDS model pipeline
Run: pytest tests/ -v
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


RANDOM_SEED = 42
N_FEATURES  = 41
CLASS_MAP   = {'Normal':0,'DoS':1,'Probe':2,'R2L':3,'U2R':4}


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate small synthetic dataset for testing."""
    rng    = np.random.RandomState(RANDOM_SEED)
    n      = 1000
    classes = list(CLASS_MAP.values())
    y = rng.choice(classes, size=n, p=[0.5,0.25,0.12,0.08,0.05])
    X = np.zeros((n, N_FEATURES))
    for i, lbl in enumerate(y):
        X[i] = rng.normal(lbl * 0.15, 0.1, N_FEATURES)
    return np.clip(X, 0, 1), y


@pytest.fixture(scope="module")
def trained_ensemble(synthetic_data):
    X, y = synthetic_data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_res, y_res = SMOTE(random_state=RANDOM_SEED, k_neighbors=3).fit_resample(X_scaled, y)

    rf  = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED, n_jobs=1)
    svc = SVC(probability=True, random_state=RANDOM_SEED)
    ens = VotingClassifier([('rf', rf), ('svc', svc)], voting='soft')
    ens.fit(X_res, y_res)
    return ens, scaler


# ─── Test: data preprocessing ────────────────────────────────────────────────

class TestPreprocessing:
    def test_scaler_range(self, synthetic_data):
        X, _ = synthetic_data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.min() >= 0.0, "MinMaxScaler: values below 0"
        assert X_scaled.max() <= 1.0, "MinMaxScaler: values above 1"

    def test_smote_balances_classes(self, synthetic_data):
        X, y = synthetic_data
        X_res, y_res = SMOTE(random_state=RANDOM_SEED, k_neighbors=3).fit_resample(X, y)
        counts = np.bincount(y_res)
        # All classes should have the same count after SMOTE
        assert counts.min() == counts.max(), "SMOTE did not balance classes"

    def test_feature_shape(self, synthetic_data):
        X, _ = synthetic_data
        assert X.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {X.shape[1]}"


# ─── Test: M1 supervised detection ───────────────────────────────────────────

class TestM1Supervised:
    def test_ensemble_predicts(self, trained_ensemble, synthetic_data):
        ens, scaler = trained_ensemble
        X, y = synthetic_data
        X_test = scaler.transform(X[:10])
        preds = ens.predict(X_test)
        assert len(preds) == 10
        assert set(preds).issubset(set(CLASS_MAP.values()))

    def test_ensemble_accuracy_above_threshold(self, trained_ensemble, synthetic_data):
        from sklearn.metrics import accuracy_score
        ens, scaler = trained_ensemble
        X, y = synthetic_data
        X_test = scaler.transform(X)
        acc = accuracy_score(y, ens.predict(X_test))
        assert acc >= 0.70, f"Accuracy {acc:.3f} is below 70% threshold"

    def test_predict_proba_sums_to_one(self, trained_ensemble, synthetic_data):
        ens, scaler = trained_ensemble
        X, _ = synthetic_data
        X_test = scaler.transform(X[:5])
        probas = ens.predict_proba(X_test)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


# ─── Test: M2 anomaly detection ───────────────────────────────────────────────

class TestM2Anomaly:
    def test_isolation_forest_trains(self, synthetic_data):
        from sklearn.ensemble import IsolationForest
        X, y = synthetic_data
        X_normal = X[y == 0]
        iforest = IsolationForest(contamination=0.05, n_estimators=10, random_state=RANDOM_SEED)
        iforest.fit(X_normal)
        scores = iforest.decision_function(X[:10])
        assert len(scores) == 10
        assert scores.dtype == float

    def test_anomaly_flags_attacks(self, synthetic_data):
        from sklearn.ensemble import IsolationForest
        X, y = synthetic_data
        X_normal = X[y == 0]
        iforest = IsolationForest(contamination=0.1, n_estimators=10, random_state=RANDOM_SEED)
        iforest.fit(X_normal)
        preds = iforest.predict(X)
        n_anomalies = (preds == -1).sum()
        assert n_anomalies > 0, "IsolationForest flagged no anomalies"


# ─── Test: M4 adversarial robustness ─────────────────────────────────────────

class TestM4Adversarial:
    def test_fgsm_perturbs_input(self, synthetic_data):
        X, _ = synthetic_data
        importance = np.ones(N_FEATURES) / N_FEATURES
        X_adv = np.clip(X + 0.05 * np.sign(importance), 0, 1)
        assert not np.allclose(X, X_adv), "FGSM did not perturb inputs"
        assert X_adv.min() >= 0 and X_adv.max() <= 1, "FGSM violated feature bounds"

    def test_feature_squeezing(self, synthetic_data):
        X, _ = synthetic_data
        bits   = 4
        levels = 2 ** bits - 1
        X_sq   = np.round(X * levels) / levels
        # Squeezed values should have limited precision
        unique_vals = len(np.unique(X_sq[:, 0]))
        assert unique_vals <= levels + 1, "Feature squeezing did not reduce precision"


# ─── Test: M5 fairness metrics ────────────────────────────────────────────────

class TestM5Fairness:
    def test_demographic_parity(self, trained_ensemble, synthetic_data):
        ens, scaler = trained_ensemble
        X, y = synthetic_data
        X_scaled = scaler.transform(X)
        y_pred   = ens.predict(X_scaled)
        # Split by binary protected attribute (feature index 1 > median)
        prot = (X[:, 1] > np.median(X[:, 1])).astype(int)

        dp0 = (y_pred[prot == 0] != 0).mean()
        dp1 = (y_pred[prot == 1] != 0).mean()
        dp_diff = abs(dp0 - dp1)

        assert dp_diff <= 0.30, (
            f"Demographic parity diff {dp_diff:.4f} exceeds 0.30 on synthetic data")
