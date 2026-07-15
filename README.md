# ET // Cyber Resilience Command Center

An AI-driven cybersecurity platform designed to protect Critical National Infrastructure (CNI) from advanced persistent threats, utilizing unsupervised anomaly detection, temporal windowing, Byzantine Fault Tolerant (BFT) agent consensus, and RAG-enriched threat intelligence.

Built for the CBSE 2026 Hackathon.

## Core Features

- **Unsupervised Anomaly Detection:** Custom `HybridAnomalyDetector` (Isolation Forest + Autoencoder) trained exclusively on normal traffic, avoiding the need for labeled attack data.
- **Calendar-Conditioning (Differentiator):** Mitigates false positives during predictable high-volume traffic bursts (e.g., exam result publications).
- **Temporal Windowing:** Rolling statistical features (mean, std, min, max) calculated per-entity to catch low-and-slow attacks like backdoors and C2 communication.
- **Multi-Agent Response with BFT:** A 3-agent Byzantine Fault Tolerance quorum ensures no single agent compromise can trigger a self-inflicted denial of service.
- **Graph AI for Lateral Movement:** NetworkX-powered attack graphs automatically map entities to identify potential pivot nodes for lateral movement.
- **RAG-Enriched Threat Intel:** TF-IDF based retrieval over a custom corpus of MITRE ATT&CK techniques, CVEs, and CERT-In advisories.
- **Human-in-the-Loop Override:** Streamlit Command Center featuring a BFT consensus log, anomaly explanation, and a strict SHA-256 hashed audit log.

---

## Setup & Installation

This project is built using standard Python data science tools. No GPUs or external API keys are required.

### 1. Prerequisites
- Python 3.10+
- (Optional but recommended) A virtual environment

### 2. Install Dependencies

```bash
# Clone the repository
git clone <repository_url>
cd ET

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Data Setup

By default, the pipeline can run using synthetic data. However, to use the real UNSW-NB15 dataset (recommended for accurate evaluation), the application will attempt to download it automatically upon first run. 

You can manually trigger the download or run the headless pipeline using:

```bash
python src/main.py
```

---

## Running the Dashboard

The primary interface is a Streamlit Command Center with a strict minimalist black-and-white aesthetic.

To launch the dashboard:

```bash
streamlit run src/app.py
```

If you don't have the real dataset downloaded and want to force synthetic data mode, run:
```bash
streamlit run src/app.py -- --synthetic
```

Navigate to `http://localhost:8501` in your browser.

---

## Running the Headless Pipeline

You can run the end-to-end evaluation pipeline via CLI to see raw metrics, ablation studies, and audit logs.

```bash
# Run with full features (BFT, Temporal Windowing, Multi-Agent) on real data:
python src/main.py --agentic --bft --temporal

# Run on synthetic data (useful for quick testing):
python src/main.py --synthetic --agentic --bft --temporal
```

**Pipeline Stages Executed:**
1. Data loading & feature engineering (including Temporal Windowing).
2. Baseline model training (No calendar conditioning).
3. Calendar-conditioned model training.
4. Calendar ablation (Measures FPR reduction on legitimate bursts).
5. MITRE ATT&CK attribution.
6. Knowledge Graph + RAG attribution enrichment.
7. SOAR + Tier-0 whitelist + SHA-256 hashed audit log.
8. Latency profiling vs. IBM 2025 Cost of a Data Breach report baseline.
9. Multi-agent coordinator pipeline (if `--agentic` is passed).

---

## Project Structure

```
ET/
├── data/                  # Dataset directory (auto-created)
├── logs/                  # Hashed SOAR audit logs and tracing
├── src/
│   ├── agents.py          # Multi-agent coordinator and specialists
│   ├── app.py             # Streamlit B&W Command Center dashboard
│   ├── bft_consensus.py   # Byzantine Fault Tolerance voting layer
│   ├── calendar_features.py # Burst-traffic conditioning
│   ├── data_loader.py     # UNSW-NB15 parser & synthetic generator
│   ├── evaluate.py        # Metrics, ablation, and tradeoff curve
│   ├── graph.py           # NetworkX attack graph & lateral movement
│   ├── knowledge_graph.py # MITRE ATT&CK kill-chain graph
│   ├── latency.py         # Latency profiling against IBM baselines
│   ├── main.py            # CLI pipeline entry point
│   ├── mitre_rag.py       # Basic attribution
│   ├── model.py           # HybridAnomalyDetector (AE + IF)
│   ├── rag_engine.py      # TF-IDF Retrieval Augmented Generation
│   ├── soar.py            # Response playbook orchestrator
│   └── temporal_features.py # Rolling window stats
├── ONBOARDING.md          # Internal project tech stack & roadmap
├── CONTEXT.md             # Project constraints & architectural choices
└── requirements.txt       # Dependencies
```
