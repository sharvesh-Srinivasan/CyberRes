"""
soar.py
Incident response orchestrator with structured playbook library.

What IS real:
  - Decision logic (playbook selection, blast-radius gating)
  - Technique-based playbook routing (ATT&CK technique ID → playbook)
  - Tier-0 criticality whitelist blocking autonomous action on CNI assets
  - SHA-256 hash-chained audit log with verify_chain()
  - Realistic playbook step definitions with API endpoint templates and payloads

What is mocked (say this to judges):
  - No actual firewall, IAM, SIEM, or EDR API calls are executed
  - api_endpoint and payload_template show what a real integration would call
  - The "response" in each step is a logged intent, not an executed action

Upgrade from v1: playbooks are now structured objects with step definitions,
and select_playbook() routes by ATT&CK technique ID rather than one-size-fits-all.
"""

import json
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_log.jsonl"

# ---------------------------------------------------------------------------
# Playbook step definition
# ---------------------------------------------------------------------------

@dataclass
class PlaybookStep:
    action: str
    target_type: str               # "endpoint" | "credential" | "network_rule" | "soc_alert"
    api_endpoint: str              # mocked — shows what a real SOAR would call
    payload_template: dict
    requires_human_approval: bool = False

    def describe(self) -> str:
        return (f"{self.action} [{self.target_type}] "
                f"→ {self.api_endpoint} "
                f"{'[HUMAN REQUIRED]' if self.requires_human_approval else '[AUTO]'}")


# ---------------------------------------------------------------------------
# Playbook library — one playbook per major attack category / technique
# ---------------------------------------------------------------------------

PLAYBOOKS: dict[str, dict] = {

    "ransomware": {
        "name": "Ransomware Containment",
        "applicable_techniques": ["T1547", "T1059", "T1210"],
        "description": "Full containment playbook for ransomware: isolate, revoke, snapshot, notify.",
        "steps": [
            PlaybookStep(
                action="flag_entity",
                target_type="endpoint",
                api_endpoint="POST /api/v1/siem/alerts",
                payload_template={"severity": "CRITICAL", "category": "ransomware",
                                  "entity_id": "{entity_id}", "risk_score": "{risk_score}"},
            ),
            PlaybookStep(
                action="isolate_endpoint",
                target_type="network_rule",
                api_endpoint="PUT /api/v1/firewall/rules/{entity_id}/isolate",
                payload_template={"action": "block_all", "direction": "both",
                                  "duration_hours": 24, "exception_ips": ["10.0.0.1"]},
            ),
            PlaybookStep(
                action="revoke_credentials",
                target_type="credential",
                api_endpoint="POST /api/v1/iam/sessions/revoke",
                payload_template={"entity_id": "{entity_id}", "scope": "all_active_sessions",
                                  "notify_user": True},
                requires_human_approval=False,
            ),
            PlaybookStep(
                action="snapshot_state",
                target_type="endpoint",
                api_endpoint="POST /api/v1/edr/snapshot",
                payload_template={"entity_id": "{entity_id}", "include_memory": True,
                                  "include_filesystem": True},
            ),
            PlaybookStep(
                action="notify_soc",
                target_type="soc_alert",
                api_endpoint="POST /api/v1/ticketing/incidents",
                payload_template={"priority": "P1", "title": "Ransomware detected: {entity_id}",
                                  "mitre_technique": "{technique_id}",
                                  "assign_to": "incident_response_team"},
            ),
        ],
    },

    "lateral_movement": {
        "name": "Lateral Movement Containment",
        "applicable_techniques": ["T1210", "T1071"],
        "description": "Targeted containment for lateral movement: isolate pivot points, alert.",
        "steps": [
            PlaybookStep(
                action="flag_entity",
                target_type="endpoint",
                api_endpoint="POST /api/v1/siem/alerts",
                payload_template={"severity": "HIGH", "category": "lateral_movement",
                                  "entity_id": "{entity_id}", "risk_score": "{risk_score}"},
            ),
            PlaybookStep(
                action="block_lateral_traffic",
                target_type="network_rule",
                api_endpoint="PUT /api/v1/firewall/rules/{entity_id}/segment",
                payload_template={"action": "block_east_west", "protocol": ["SMB", "RDP", "SSH"],
                                  "allow_outbound_dns": True},
            ),
            PlaybookStep(
                action="snapshot_state",
                target_type="endpoint",
                api_endpoint="POST /api/v1/edr/snapshot",
                payload_template={"entity_id": "{entity_id}", "include_network_connections": True},
            ),
            PlaybookStep(
                action="notify_soc",
                target_type="soc_alert",
                api_endpoint="POST /api/v1/ticketing/incidents",
                payload_template={"priority": "P2", "title": "Lateral movement: {entity_id}",
                                  "mitre_technique": "{technique_id}"},
            ),
        ],
    },

    "dos_attack": {
        "name": "DoS / DDoS Mitigation",
        "applicable_techniques": ["T1498"],
        "description": "Rate-limit and upstream mitigation for denial-of-service events.",
        "steps": [
            PlaybookStep(
                action="flag_entity",
                target_type="endpoint",
                api_endpoint="POST /api/v1/siem/alerts",
                payload_template={"severity": "HIGH", "category": "dos",
                                  "entity_id": "{entity_id}", "risk_score": "{risk_score}"},
            ),
            PlaybookStep(
                action="rate_limit_source",
                target_type="network_rule",
                api_endpoint="POST /api/v1/ddos-protection/rules",
                payload_template={"source_ip": "{entity_id}", "action": "rate_limit",
                                  "packets_per_second_limit": 100,
                                  "duration_minutes": 60},
            ),
            PlaybookStep(
                action="notify_soc",
                target_type="soc_alert",
                api_endpoint="POST /api/v1/ticketing/incidents",
                payload_template={"priority": "P2", "title": "DoS attack detected: {entity_id}",
                                  "mitre_technique": "{technique_id}"},
            ),
        ],
    },

    "reconnaissance": {
        "name": "Reconnaissance Monitoring",
        "applicable_techniques": ["T1595", "T1592"],
        "description": "Passive monitoring and honey-token enrichment for recon activity.",
        "steps": [
            PlaybookStep(
                action="flag_entity",
                target_type="endpoint",
                api_endpoint="POST /api/v1/siem/alerts",
                payload_template={"severity": "MEDIUM", "category": "reconnaissance",
                                  "entity_id": "{entity_id}", "risk_score": "{risk_score}"},
            ),
            PlaybookStep(
                action="enable_enhanced_logging",
                target_type="network_rule",
                api_endpoint="PUT /api/v1/ids/rules/{entity_id}",
                payload_template={"log_level": "VERBOSE", "capture_full_packet": False,
                                  "alert_on_new_dst_port": True},
            ),
            PlaybookStep(
                action="notify_soc",
                target_type="soc_alert",
                api_endpoint="POST /api/v1/ticketing/incidents",
                payload_template={"priority": "P3", "title": "Recon detected: {entity_id}",
                                  "mitre_technique": "{technique_id}"},
            ),
        ],
    },

    "initial_access": {
        "name": "Initial Access Response",
        "applicable_techniques": ["T1190"],
        "description": "Block and patch response for exploitation of public-facing applications.",
        "steps": [
            PlaybookStep(
                action="flag_entity",
                target_type="endpoint",
                api_endpoint="POST /api/v1/siem/alerts",
                payload_template={"severity": "CRITICAL", "category": "initial_access",
                                  "entity_id": "{entity_id}", "risk_score": "{risk_score}"},
            ),
            PlaybookStep(
                action="block_source_ip",
                target_type="network_rule",
                api_endpoint="POST /api/v1/waf/block",
                payload_template={"source_ip": "{entity_id}", "duration_hours": 48,
                                  "apply_to_zones": ["dmz", "perimeter"]},
            ),
            PlaybookStep(
                action="isolate_endpoint",
                target_type="network_rule",
                api_endpoint="PUT /api/v1/firewall/rules/{entity_id}/isolate",
                payload_template={"action": "block_inbound", "preserve_management_access": True},
                requires_human_approval=True,
            ),
            PlaybookStep(
                action="snapshot_state",
                target_type="endpoint",
                api_endpoint="POST /api/v1/edr/snapshot",
                payload_template={"entity_id": "{entity_id}", "include_web_logs": True},
            ),
            PlaybookStep(
                action="notify_soc",
                target_type="soc_alert",
                api_endpoint="POST /api/v1/ticketing/incidents",
                payload_template={"priority": "P1", "title": "Exploitation attempt: {entity_id}",
                                  "mitre_technique": "{technique_id}",
                                  "recommend_patch": True},
            ),
        ],
    },

    "default": {
        "name": "General Anomaly Response",
        "applicable_techniques": [],
        "description": "Catch-all playbook for unclassified anomalies.",
        "steps": [
            PlaybookStep(
                action="flag_entity",
                target_type="endpoint",
                api_endpoint="POST /api/v1/siem/alerts",
                payload_template={"severity": "MEDIUM", "category": "anomaly",
                                  "entity_id": "{entity_id}", "risk_score": "{risk_score}"},
            ),
            PlaybookStep(
                action="notify_soc",
                target_type="soc_alert",
                api_endpoint="POST /api/v1/ticketing/incidents",
                payload_template={"priority": "P3", "title": "Anomaly detected: {entity_id}",
                                  "mitre_technique": "{technique_id}"},
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Technique ID → playbook key routing table
# ---------------------------------------------------------------------------
_TECHNIQUE_TO_PLAYBOOK: dict[str, str] = {
    "T1595": "reconnaissance",
    "T1592": "reconnaissance",
    "T1190": "initial_access",
    "T1059": "ransomware",
    "T1547": "ransomware",
    "T1498": "dos_attack",
    "T1071": "lateral_movement",
    "T1210": "lateral_movement",
}


def select_playbook(technique_id: Optional[str]) -> tuple[str, dict]:
    """Select and return (playbook_key, playbook_dict) for an ATT&CK technique ID."""
    key = _TECHNIQUE_TO_PLAYBOOK.get(technique_id or "", "default")
    return key, PLAYBOOKS[key]


# ---------------------------------------------------------------------------
# Tier-0 whitelist and thresholds
# ---------------------------------------------------------------------------

TIER0_WHITELIST = {
    "10.0.0.1",   # core DNS
    "10.0.0.2",   # domain controller
    "10.0.5.10",  # exam-results database (education sector Tier-0 example)
    "10.0.9.1",   # life-support / OT gateway (hospital example)
}

RISK_THRESHOLD_AUTO_ACTION = 0.85
RISK_THRESHOLD_ESCALATE = 0.70


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@dataclass
class AuditLog:
    entries: list = field(default_factory=list)
    prev_hash: str = "0" * 64  # genesis hash

    def record(self, event: dict) -> dict:
        payload = json.dumps(event, sort_keys=True, default=str)
        entry_hash = hashlib.sha256((self.prev_hash + payload).encode()).hexdigest()
        record = {**event, "prev_hash": self.prev_hash, "entry_hash": entry_hash,
                   "timestamp": time.time()}
        self.entries.append(record)
        self.prev_hash = entry_hash
        return record

    def verify_chain(self) -> bool:
        """Recompute the hash chain to prove no log entry was tampered with post-hoc."""
        prev = "0" * 64
        for e in self.entries:
            payload = json.dumps({k: v for k, v in e.items()
                                   if k not in ("prev_hash", "entry_hash", "timestamp")},
                                  sort_keys=True, default=str)
            expected = hashlib.sha256((prev + payload).encode()).hexdigest()
            if expected != e["entry_hash"]:
                return False
            prev = e["entry_hash"]
        return True

    def persist(self, path: Path = LOG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for e in self.entries:
                f.write(json.dumps(e, default=str) + "\n")


# ---------------------------------------------------------------------------
# Response Orchestrator
# ---------------------------------------------------------------------------

class ResponseOrchestrator:
    def __init__(self, audit_log: AuditLog = None):
        self.audit_log = audit_log or AuditLog()

    def handle(self, entity_id: str, risk_score: float,
               mitre_technique: str = None,
               consensus_result=None,
               rag_evidence: dict = None,
               top_anomaly_features: list = None) -> dict:
        """
        Decide on and log a SOAR response for an entity.

        Args:
            entity_id: IP or identifier string.
            risk_score: Anomaly risk score [0, 1].
            mitre_technique: ATT&CK technique ID for playbook routing.
            consensus_result: Optional ConsensusResult from BFTConsensusLayer.
                When provided, its recommended_decision overrides the
                risk-band routing (BFT consensus takes precedence).
        """
        is_tier0 = entity_id in TIER0_WHITELIST
        playbook_key, playbook = select_playbook(mitre_technique)

        # --- Decision routing ---
        if consensus_result is not None:
            # BFT consensus overrides risk-band routing
            bft_decision = consensus_result.recommended_decision
            if bft_decision == "ESCALATE_ONLY" or is_tier0:
                decision = "ESCALATE_ONLY"
                steps_executed = ["flag_entity", "notify_soc"]
                reason = "Tier-0 asset or BFT consensus: autonomous action forbidden"
            elif bft_decision == "AUTO_CONTAIN":
                decision = "AUTO_CONTAIN"
                steps_executed = [s.action for s in playbook["steps"]]
                reason = (f"BFT consensus FLAGGED ({consensus_result.vote_count}/3 agents), "
                          f"risk_mean={consensus_result.risk_score_mean:.2f}")
            elif bft_decision == "ESCALATE":
                decision = "ESCALATE"
                steps_executed = [s.action for s in playbook["steps"]
                                  if not s.requires_human_approval]
                reason = (f"BFT consensus DISPUTED ({consensus_result.vote_count}/3 agents) — "
                          f"human review required")
            else:  # CLEARED / MONITOR
                decision = "MONITOR"
                steps_executed = ["flag_entity"]
                reason = f"BFT consensus CLEARED ({consensus_result.vote_count}/3 agents flagged)"
        else:
            # Standard risk-band routing (no BFT)
            if is_tier0:
                decision = "ESCALATE_ONLY"
                steps_executed = ["flag_entity", "notify_soc"]
                reason = "Tier-0 asset: autonomous action forbidden regardless of risk score"
            elif risk_score >= RISK_THRESHOLD_AUTO_ACTION:
                decision = "AUTO_CONTAIN"
                steps_executed = [s.action for s in playbook["steps"]]
                reason = (f"risk_score {risk_score:.2f} >= auto-action threshold "
                          f"{RISK_THRESHOLD_AUTO_ACTION}")
            elif risk_score >= RISK_THRESHOLD_ESCALATE:
                decision = "ESCALATE"
                steps_executed = [s.action for s in playbook["steps"]
                                  if not s.requires_human_approval]
                reason = (f"risk_score {risk_score:.2f} in escalation band "
                          f"[{RISK_THRESHOLD_ESCALATE}, {RISK_THRESHOLD_AUTO_ACTION})")
            else:
                decision = "MONITOR"
                steps_executed = ["flag_entity"]
                reason = f"risk_score {risk_score:.2f} below escalation threshold"

        event = {
            "entity_id": entity_id,
            "risk_score": risk_score,
            "mitre_technique": mitre_technique,
            "playbook_key": playbook_key,
            "playbook_name": playbook["name"],
            "decision": decision,
            "steps_executed": steps_executed,
            "automation_coverage": f"{len(steps_executed)}/{len(playbook['steps'])}",
            "reason": reason,
            "is_tier0": is_tier0,
            "bft_consensus": consensus_result.to_dict() if consensus_result else None,
            # RAG evidence (Weakness 3 fix): what threat intel backs this decision
            "rag_evidence": {
                "technique": rag_evidence.get("technique_doc", {}).get("title") if rag_evidence else None,
                "cve_count":  len(rag_evidence.get("cve_evidence", [])) if rag_evidence else 0,
                "advisory":   rag_evidence.get("advisory_evidence", {}).get("title") if rag_evidence else None,
            } if rag_evidence else None,
            # Top anomaly features (Weakness 6 fix): WHY this entity was flagged
            "top_anomaly_features": [
                {"feature": f["feature"], "contribution": round(f["normalized_contribution"], 3)}
                for f in (top_anomaly_features or [])[:3]
            ] or None,
        }
        return self.audit_log.record(event)
