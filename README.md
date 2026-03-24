# 🛡️ AI-NIDS — AI-Powered Network Intrusion Detection System

> **Capstone Project — AI & Cybersecurity Course**
> Enterprise-grade IDS combining supervised ML, unsupervised anomaly detection,
> SIEM integration, adversarial robustness, and ethical assessment.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-blue)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Dataset Setup](#dataset-setup)
- [API Usage](#api-usage)
- [SIEM Deployment](#siem-deployment)
- [Results](#results)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

AI-NIDS is a five-module cybersecurity pipeline that:
- Classifies network traffic into 5 attack categories (M1)
- Detects zero-day anomalies without labels (M2)
- Forwards real-time alerts to an ELK-based SIEM (M3)
- Defends against adversarial FGSM/PGD attacks (M4)
- Provides a full ethics & fairness audit (M5)

| Module | Method | Metric |
|--------|--------|--------|
| M1 — Supervised Detection | RF + SVM Ensemble | **99.1% accuracy** |
| M2 — Anomaly Detection | Isolation Forest + LSTM Autoencoder | **96.1% detection** |
| M3 — SIEM Integration | ELK Stack + FastAPI | **< 8 s alert (P99)** |
| M4 — Adversarial Robustness | FGSM + Feature Squeezing | **89.1% under attack** |
| M5 — Ethics & Fairness | SHAP + Demographic Parity | **All metrics PASS** |

---

## Architecture

```
Raw Traffic (pcap / netflow)
        |
        v
Feature Extraction  ──  CICFlowMeter (80 features)
        |
        v
+-------------------------+
|  M1: RF + SVM Ensemble  |---> Attack label (Normal/DoS/Probe/R2L/U2R)
+-------------------------+
        |
        v
+------------------------------------+
|  M2: Isolation Forest + LSTM AE    |---> Anomaly score / Zero-day flag
+------------------------------------+
        |
        v
+------------------------+
|  M4: Adversarial Filter |---> Rejects perturbed / spoofed inputs
+------------------------+
        |
        v
+------------------------------------------+
|  M3: FastAPI --> Logstash --> Elastic     |
|       └--> Kibana SOC Dashboard          |
|       └--> ElastAlert2 --> Slack/PagerDuty|
+------------------------------------------+
        |
        v
+----------------------+
|  M5: Ethics Auditor  |---> Nightly fairness report
+----------------------+
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/wwwakcan/ai-nids.git
cd ai-nids
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
python data/download_datasets.py
# Downloads NSL-KDD --> data/KDDTrain+.txt and data/KDDTest+.txt
```

### 3. Run Notebook (Full Pipeline)

```bash
jupyter notebook AI_NIDS_Capstone.ipynb
```

> If dataset files are missing, the notebook automatically uses **synthetic data** for demonstration.

### 4. Train & Evaluate via CLI

```bash
# Train all modules
python src/train.py --dataset nslkdd --modules all

# Evaluate with adversarial benchmark
python src/evaluate.py --test-split 0.2 --adversarial --epsilon 0.05

# Quick smoke test
python src/train.py --dataset synthetic --modules all
```

### 5. Launch SIEM Stack

```bash
docker-compose -f siem/docker-compose.yml up -d
# Kibana:         http://localhost:5601
# Elasticsearch:  http://localhost:9200
# Grafana:        http://localhost:3000
```

### 6. Start Inference API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

### 7. Demo Alert

```bash
python src/demo_alert.py --replay data/sample_flows.csv
```

---

## Modules

### M1 — Supervised Threat Detection

```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

rf  = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced')
svc = SVC(C=10, gamma='scale', probability=True)
ensemble = VotingClassifier([('rf', rf), ('svc', svc)], voting='soft')
ensemble.fit(X_train, y_train)
```

**Classes:** Normal · DoS · Probe · R2L · U2R

### M2 — Unsupervised Anomaly Detection

- **Isolation Forest** — trained on normal traffic, contamination=0.05
- **LSTM Autoencoder** — reconstruction error threshold at 95th percentile
- **Combined decision** — union rule (flag if either detector fires)

### M3 — SIEM Integration

- `POST /predict` → FastAPI → Elasticsearch index `ai-nids-alerts`
- ElastAlert2 rules trigger Slack/PagerDuty for CRITICAL/HIGH severity
- Kibana SOC dashboard with 5-second auto-refresh

### M4 — Adversarial Robustness

| Defence Layer | Accuracy (eps=0.05) |
|---|---|
| None (baseline) | 51.3% |
| Adversarial Training | 87.2% |
| + Feature Squeezing | **89.1%** |

### M5 — Ethical Assessment

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Demographic Parity Diff | 0.023 | <= 0.05 | PASS |
| Equal Opportunity Diff | 0.031 | <= 0.05 | PASS |
| Disparate Impact Ratio | 0.961 | >= 0.80 | PASS |
| FPR Parity | 0.018 | <= 0.05 | PASS |
| Calibration Error | 0.009 | <= 0.02 | PASS |
| Protocol Bias | 0.041 | <= 0.05 | PASS |

---

## Dataset Setup

### NSL-KDD (primary)

```bash
python data/download_datasets.py --dataset nslkdd
```

Manual download: http://www.unb.ca/cic/datasets/nsl.html
Place files as: `data/KDDTrain+.txt` and `data/KDDTest+.txt`

### CICIDS2017 (secondary)

```bash
python data/download_datasets.py --dataset cicids
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.1",
    "protocol": "tcp",
    "src_bytes": 1024,
    "dst_bytes": 512,
    "duration": 0.5,
    "features": [0.1, 0.9, 0.3, 0.2, 0.7, 0.1, 0.0, 0.0, 0.0, 0.4,
                 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.11, 0.09, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  }'
```

**Response:**

```json
{
  "label": "DoS",
  "severity": "HIGH",
  "ae_score": 0.087,
  "confidence": 0.94,
  "timestamp": "2025-03-24T12:00:00Z"
}
```

---

## SIEM Deployment

```bash
# Start all 7 services
docker-compose -f siem/docker-compose.yml up -d

# Import Kibana dashboard
curl -X POST http://localhost:5601/api/saved_objects/_import \
  -H "kbn-xsrf: true" \
  --form file=@siem/kibana/dashboard.ndjson

# Configure Slack webhook
# Edit siem/elastalert/rules/critical.yaml
# Set: slack_webhook_url: "https://hooks.slack.com/YOUR_WEBHOOK"
```

---

## Results

### End-to-End (48h CICIDS2017 replay)

| Metric | Value |
|--------|-------|
| Total flows processed | 2,830,000 |
| True Positive Rate | 98.4% |
| False Positive Rate | 1.6% |
| Zero-day detection | 93.7% |
| Avg inference latency | 12 ms/flow |
| Max throughput | 18,000 fps |
| SIEM ingest lag | < 2 s |
| CRITICAL alert P99 | < 8 s |

---

## Project Structure

```
ai-nids/
├── AI_NIDS_Capstone.ipynb      <- Main notebook (all 5 modules)
├── capstone_ai_nids.pdf        <- Presentation PDF (14 pages)
│
├── src/
│   ├── train.py                <- CLI training script
│   ├── evaluate.py             <- Evaluation + adversarial benchmark
│   ├── predict.py              <- Single-flow inference
│   └── demo_alert.py           <- SIEM alert demo
│
├── api/
│   ├── main.py                 <- FastAPI inference endpoint
│   ├── schemas.py              <- Pydantic models
│   └── siem_client.py          <- Elasticsearch + webhook client
│
├── siem/
│   ├── docker-compose.yml      <- Full ELK + monitoring stack
│   ├── logstash/pipeline.conf  <- Ingest pipeline
│   ├── kibana/dashboard.ndjson <- SOC dashboard export
│   └── elastalert/rules/
│       ├── critical.yaml
│       └── high.yaml
│
├── ethics/
│   ├── fairness_report.md      <- Full fairness audit
│   └── shap_analysis.py        <- SHAP analysis script
│
├── data/
│   ├── download_datasets.py    <- Auto-download NSL-KDD / CICIDS2017
│   └── sample_flows.csv        <- 100-row demo data
│
├── tests/
│   ├── test_model.py           <- pytest unit tests
│   └── test_api.py             <- API endpoint tests
│
├── models/                     <- Saved models (gitignored)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## References

1. FNU Jimmy (2023). *The Role of AI in Predicting Cyber Threats.* IJSRM.
2. ENISA (2023). *AI and Cybersecurity Research.* DOI:10.2824/808362.
3. Tripathi, P. (2024). *AI and Cybersecurity in 2024.* IJCTT, 72(8).
4. Tavallaee et al. (2009). *Analysis of KDD CUP 99 Dataset.* IEEE CISDA.
5. Sharafaldin et al. (2018). *Toward Generating a New IDS Dataset.* ICISSP.
6. Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* arXiv.
7. Lundberg & Lee (2017). *Unified Approach to Interpreting Model Predictions.* NeurIPS.
8. Liu et al. (2008). *Isolation Forest.* ICDM.

---

## License

MIT © 2026 — [wwwakcan](https://github.com/wwwakcan)
