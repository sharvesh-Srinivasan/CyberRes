# Pitch Narrative: AI-Driven Cyber Resilience

This document outlines the complete narrative of the project, structured exactly how it should be pitched to the judges—from the core problem statement through to the final architecture.

---

## 1. The Problem Statement: Why Existing Defenses Fail

Critical National Infrastructure (CNI)—such as government networks, educational institutions, and healthcare systems—is increasingly targeted by advanced persistent threats (APTs). 

When defending these networks, Security Operations Centers (SOCs) rely on User and Entity Behavior Analytics (UEBA) to detect anomalies. However, standard anomaly detectors fail in a very specific, painful way: **they cannot distinguish between an attack and a legitimate institutional traffic burst.** 

For example, when the CBSE publishes board exam results, or a university opens admissions, network traffic spikes massively. A standard anomaly detector flags this as a volumetric attack (like a DDoS or data exfiltration). This triggers thousands of false positives. SOC analysts get alert fatigue, start ignoring the dashboard, and when a real attack happens (like the AIIMS ransomware in 2022 or the CBSE breach in 2024), it slips through the noise.

**The core problem we are solving:** How do we build an autonomous cyber resilience system that catches zero-day attacks without drowning analysts in false positives during predictable, high-volume institutional events?

---

## 2. Our Core Solution: The Calendar-Conditioned Differentiator

Instead of just throwing a bigger neural network at the problem, we solved the false-positive issue through **Calendar-Conditioned Baselining**.

We built a **Hybrid Anomaly Detector** (an Autoencoder feeding into an Isolation Forest). We trained it *strictly* on normal traffic (unsupervised learning) so it can catch novel, zero-day attacks without needing labeled threat data. 

But crucially, we engineered the model to be aware of the institution's calendar. By feeding it synthetic "burst" data tagged with calendar phases (e.g., `exam_period`, `admission_season`), the model learns that a massive spike in traffic is *normal* if it aligns with the academic or fiscal calendar.

**The Result:** On legitimate traffic bursts, our calendar-conditioning **reduced the false-positive rate by 46.5%** (on synthetic data) and **17.9%** (on real UNSW-NB15 data). We eliminate the noise so the SOC can focus on actual threats.

---

## 3. The Full Pipeline: What We Built

To make this a complete, enterprise-grade Cyber Resilience system, we wrapped the core anomaly detector in a sophisticated, multi-stage pipeline:

### A. Temporal Windowing (Catching Low-and-Slow Attacks)
Hackers know how to evade point-in-time anomaly detection by acting slowly (e.g., dripping data out over weeks). We implemented **Temporal Windowing**, which calculates rolling statistics (mean, min, max, std) for every single IP address over time. This gives the model memory, allowing it to spot subtle backdoors and command-and-control (C2) beacons that would otherwise blend in.

### B. Graph AI & Lateral Movement
When an entity is flagged as anomalous, we don't just look at it in isolation. We automatically map it onto a **NetworkX Attack Graph**. This graph traces the connections between IPs, allowing us to detect **lateral movement paths**—showing exactly which adjacent servers are at risk of being compromised next so the SOC can cut off the pivot nodes.

### C. Multi-Agent System & BFT Consensus
To orchestrate the response, we built an Agentic AI coordinator with three specialist agents (Detection, Attribution, and Response). 
Crucially, before the system is allowed to automatically isolate a server, it must pass a **Byzantine Fault Tolerance (BFT) Consensus Gate**. Three independent detection agents vote on the risk. An automated firewall block only occurs if a 2/3 quorum agrees. If the agents dispute the threat, it is safely escalated to a human. This guarantees that a single compromised agent cannot be weaponized to cause a self-inflicted Denial of Service.

### D. RAG-Enriched Threat Intelligence & Knowledge Graphs
Once an attack is detected, our pipeline maps it to a **MITRE ATT&CK Knowledge Graph** to figure out exactly where the attacker is in the kill-chain (e.g., "Phase 10/14: Lateral Movement"). 
We then use **Retrieval-Augmented Generation (RAG)** using TF-IDF over a custom corpus of MITRE techniques, CVEs, and CERT-In advisories to pull the exact documentation the analyst needs to understand the threat.

### E. SOAR Orchestration & Immutable Audit Log
Finally, the decision (e.g., "Auto-Contain" or "Escalate") is routed through our Security Orchestration, Automation, and Response (SOAR) logic. 
Because autonomous systems in critical infrastructure must be auditable, every decision, the RAG evidence backing it, and the top 3 features that caused the anomaly are written to a **SHA-256 hash-chained Audit Log**. If a decision is ever questioned, the cryptographically secure chain proves exactly why the AI took action.

---

## 4. The Final Dashboard (Command Center)

All of this is surfaced in a strict, minimalist **Black-and-White Streamlit Command Center**. It isn't a toy dashboard; it is designed for a real SOC analyst. It features:
- Live metrics showing the FPR-Recall tradeoff curve.
- The BFT consensus log showing exactly how the agents voted.
- A Human-in-the-Loop override gate to approve or dismiss disputed actions.
- The NetworkX lateral movement graph.
- The full, verifiable SHA-256 audit log.

---

## Summary for the Judges
We aren't just presenting a generic AI classifier. We built a domain-aware resilience pipeline that reduces alert fatigue via calendar-conditioning, catches advanced threats via temporal windowing, prevents autonomous self-harm via BFT agent consensus, and provides cryptographic auditability for every action taken on Critical National Infrastructure.
