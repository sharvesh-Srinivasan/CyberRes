"""
mitre_rag.py
Attribution layer: maps flagged anomalies to MITRE ATT&CK techniques.

Honest scope note: this is a curated lookup table (grounded in real
UNSW-NB15 category -> ATT&CK technique mappings), not a live RAG over a
vector database of the full ATT&CK corpus. For the hackathon deliverable
this is defensible IF you say so explicitly -- it demonstrates the
attribution *logic* correctly; swapping the dict lookup for a real
embedding-indexed vector store (MITRE STIX corpus + CERT-In advisories)
is a documented "next step", not a claim we make about this build.
"""

# UNSW-NB15 attack_cat -> MITRE ATT&CK technique(s), chosen for a
# defensible mapping (not exhaustive -- each category maps to its most
# representative technique for a first-pass attribution).
CATEGORY_TO_TECHNIQUE = {
    "Fuzzers":        {"id": "T1595", "name": "Active Scanning", "tactic": "Reconnaissance"},
    "Analysis":        {"id": "T1592", "name": "Gather Victim Host Information", "tactic": "Reconnaissance"},
    "Backdoor":       {"id": "T1547", "name": "Boot or Logon Autostart Execution", "tactic": "Persistence"},
    "DoS":             {"id": "T1498", "name": "Network Denial of Service", "tactic": "Impact"},
    "Exploits":        {"id": "T1190", "name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
    "Generic":         {"id": "T1071", "name": "Application Layer Protocol", "tactic": "Command and Control"},
    "Reconnaissance":  {"id": "T1595", "name": "Active Scanning", "tactic": "Reconnaissance"},
    "Shellcode":       {"id": "T1059", "name": "Command and Scripting Interpreter", "tactic": "Execution"},
    "Worms":           {"id": "T1210", "name": "Exploitation of Remote Services", "tactic": "Lateral Movement"},
    "Normal":          None,
}


def attribute(attack_cat: str) -> dict | None:
    """Given a predicted/observed category label, return the MITRE technique attribution."""
    return CATEGORY_TO_TECHNIQUE.get(attack_cat, {
        "id": "UNKNOWN", "name": "Unclassified anomalous behavior", "tactic": "Unknown"
    })


def attribution_accuracy(predicted_cats: list[str], true_cats: list[str]) -> float:
    """
    Reports technique-level attribution accuracy: fraction of samples where
    the technique mapped from the PREDICTED category matches the technique
    mapped from the TRUE category. This is the number to report for the
    "APT attribution accuracy" evaluation metric.
    """
    correct = 0
    total = 0
    for pred, true in zip(predicted_cats, true_cats):
        pred_attr = attribute(pred)
        true_attr = attribute(true)
        if true_attr is None:
            continue  # skip normal traffic, attribution only scored on actual attacks
        total += 1
        if pred_attr is not None and pred_attr.get("id") == true_attr.get("id"):
            correct += 1
    return correct / total if total else 0.0
