# Onboarding — AI-Driven Cyber Resilience for Critical National Infrastructure

Read this first. It's the fast path to being productive on this repo. For full
depth on every design decision, read `CONTEXT.md` next — this file is the map,
that file is the territory.

---

## 1. What this project is

A hackathon prototype for AI-driven cyber resilience for Indian critical
infrastructure (government/education/healthcare). Framed around real incidents
(AIIMS ransomware 2022, CBSE breaches 2024/early 2026).

**Judging weights:** Business Impact 25% · Technical Excellence 25% ·
Scalability 20% · User Experience 15% · Innovation 15%.

**Suggested technologies from the brief (honest status of each):**

| Technology | Status |
|---|---|
| Unsupervised Anomaly Detection (UEBA) | ✅ **Real** — per-entity Autoencoder→IsolationForest, trained on normal-only traffic, calibrated thresholds. |
| Temporal Windowing | ✅ **Real** — rolling statistics calculated per-entity to catch low-and-slow attacks (Fix 4). |
| Agentic AI / Multi-Agent Systems | ✅ **Real** — Coordinator + Detection, Attribution, and Response agents (`src/agents.py`). |
| BFT Consensus | ✅ **Real** — 3-agent 2/3 quorum voting before auto-containment (`src/bft_consensus.py`). |
| Graph AI (attack path / lateral movement) | ✅ **Real** — NetworkX entity graph detecting pivot nodes (`src/graph.py`). |
| RAG over threat intelligence / CVE / CERT-In | ✅ **Real** — TF-IDF retrieval over custom text corpus of MITRE/CVE/CERT-In docs (`src/rag_engine.py`). |
| Knowledge Graphs (MITRE ATT&CK TTP mapping) | ✅ **Real** — NetworkX kill-chain graph mapping tactics to 14 phases (`src/knowledge_graph.py`). |
| SOAR Integration & Response Automation | ⚠️ **Partial** — decision logic (playbooks, Tier-0 gating, BFT gates) and SHA-256 audit log are real. Zero real API calls — actions are logged intent only. |

**What we built well:** The anomaly detector + calendar-conditioning differentiator is the one genuinely hard ML thing. The multi-agent BFT consensus and RAG attribution pipeline (Phase 2) provide the architecture that wins hackathons.

---

## 2. Non-negotiable ground rule: real numbers only

Every metric that goes in the deck, README, or a resume bullet must come
from a run against the **real UNSW-NB15 dataset**, not synthetic smoke-test
data. `--synthetic` mode exists only to verify the pipeline runs without
crashing.

**Real benchmark numbers (2026-07-15 run, 82,332 test rows):**

| Metric | Baseline | Calendar-Conditioned |
|---|---|---|
| Precision | 0.774 | **0.804** |
| Recall | 0.325 | **0.371** |
| Overall FPR | 0.116 | **0.111** |
| Burst-traffic FPR reduction | — | **17.9%** |
| p95 detect+decide latency | — | **36 ms** |

**Per-category recall (calendar-conditioned):**

| Category | Recall | n |
|---|---|---|
| Exploits | 0.537 | 11,132 |
| Generic | 0.332 | 18,871 |
| Worms | 0.318 | 44 |
| DoS | 0.215 | 4,089 |
| Reconnaissance | 0.196 | 3,496 |
| Shellcode | 0.153 | 378 |
| Fuzzers | 0.133 | 6,062 |
| Backdoor | 0.072 | 583 |
| Analysis | 0.030 | 677 |

Getting the real data (no Kaggle auth needed):
```bash
curl -sL -o data/UNSW_NB15_training-set.csv "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
curl -sL -o data/UNSW_NB15_testing-set.csv "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"
```
Then `python src/main.py` (no flag = real data).

---

## 3. What's real vs. mocked — say this explicitly to judges

| Component | Status |
|---|---|
| Autoencoder + Isolation Forest hybrid detector | ✅ Real, trained, evaluated |
| Calendar-phase conditioning (exam/admission/fiscal seasons) | ✅ Real mechanism; phases themselves are synthetic (UNSW-NB15 has no usable real timestamps) |
| Per-attack-category precision/recall/FPR | ✅ Real, computed from confusion matrix |
| Tier-0 criticality whitelist | ✅ Real, hardcoded, enforced in code |
| SHA-256 hash-chained audit log | ✅ Real, verifiable via `AuditLog.verify_chain()` |
| MTTD/MTTR latency measurement | ✅ Real inference + SOAR timing vs. IBM 2025 baseline |
| Graph / lateral movement detection | 🔨 In progress (`src/graph.py`) |
| Streamlit dashboard | 🔨 In progress (`src/app.py`) |
| MITRE ATT&CK attribution | ⚠️ Real logic, curated lookup — **not** a live vector-DB RAG |
| SOAR playbook execution | ⚠️ Decision logic real; no actual firewall/IAM API calls |
| Kafka ingestion, OT protocol parsing, LSTM temporal model | ❌ Not built — architecture-diagram only, documented future work |

---

## 4. File map

```
src/
  data_loader.py        # UNSW-NB15 loading + synthetic fallback
  calendar_features.py  # calendar-phase conditioning (the differentiator)
  model.py              # HybridAnomalyDetector: AE latent -> Isolation Forest
  evaluate.py           # per-category metrics + calendar ablation
  mitre_rag.py          # MITRE ATT&CK attribution lookup
  soar.py               # mock SOAR + Tier-0 whitelist + hashed audit log
  latency.py            # MTTD/MTTR latency measurement vs. IBM baseline
  graph.py              # [IN PROGRESS] NetworkX attack graph + lateral movement
  app.py                # [IN PROGRESS] Streamlit dashboard
  main.py               # orchestrates the full pipeline end-to-end

data/                   # gitignored — place UNSW-NB15 CSVs here
logs/                   # audit_log.jsonl written here at runtime
```

One file, one job. A new top-level file means a new pipeline stage — that's a
bigger decision than most changes warrant.

---

## 5. Known bugs already found and fixed (don't reintroduce these)

1. **Score normalization was batch-relative, not calibrated.** `model.py`
   `score()` used to compute min/max off the inference batch, not a fixed range
   from training. Looked fine on synthetic data, silently broke on real imbalanced
   data (recall collapsed to ~0.1%). Fixed: calibration ranges and threshold are
   now derived from training-normal distribution at fit time (`calibrate_threshold()`).
2. **`pd.get_dummies` returns `bool` dtype** in current pandas, silently dropped
   by `select_dtypes(include=[np.number])`. `proto`/`service`/`state` features
   never reached the model. Fixed: cast dummy columns to `int8` before filter.
3. **Category name mismatch:** real dataset uses `"Backdoor"` (singular). Code had
   `"Backdoors"` everywhere. Fixed consistently across `data_loader.py` and
   `mitre_rag.py`.
4. `ndarray.ptp()` removed in NumPy 2.x — use `np.ptp(array)`.

If recall looks near-zero on real data, check calibration bug hasn't crept back.

---

## 6. Known limitations — say these to judges, don't hide them

| Limitation | Honest framing |
|---|---|
| Point-in-time only, no temporal modeling | Low-and-slow multi-day APTs would likely evade detection. Windowed features are the cheap fix; LSTM-Autoencoder is the expensive one (future work). |
| Low recall on stealthy categories (Backdoor 7%, Analysis 3%) | Expected — unsupervised detector has no attack signatures. Stealthy categories are designed to be behaviorally indistinct from normal. |
| Calendar phases are synthetic | UNSW-NB15 has no usable real timestamps. Phase assignment would be a real calendar lookup in production. |
| SOAR actions are not real | Decision logic and audit log are real. Isolation/revocation are logged intent only. |
| Attribution is a lookup table, not RAG | Correct MITRE mappings, but not retrieved from a live vector store. |

---

## 7. Before you push

- Never commit result numbers to the deck/README without confirming they came
  from a non-`--synthetic` run.
- Never commit `data/` CSVs to git (large file, `.gitignore` should exclude `data/*.csv`).
- If you touch `model.py` scoring or threshold logic, rerun `python src/main.py`
  end-to-end and check per-category recall isn't near-zero before pushing.
- If you touch `graph.py` or `app.py`, run `streamlit run src/app.py` and confirm
  the dashboard loads with both synthetic and real data.

---

## 8. Where to go deeper

- `CONTEXT.md` — full history of every scope decision and why (what was proposed,
  adopted, explicitly rejected, and what broke on first real-data run).
- `README.md` — setup + run instructions.
