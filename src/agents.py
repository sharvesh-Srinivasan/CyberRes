"""
agents.py
Multi-agent coordinator for AI-driven cyber resilience.

Architecture (genuine agentic pattern, not marketing):
  CyberResilienceCoordinator
  ├── DetectionAgent        → wraps HybridAnomalyDetector, scores entities
  ├── AttributionAgent      → wraps RAGEngine + MITREKnowledgeGraph, enriches detections
  └── ResponseAgent         → wraps ResponseOrchestrator + graph lateral-movement check

Each agent has a defined role, capabilities, and a run(context) -> AgentResult method.
Agents communicate via a shared AgentContext message bus (dict, no network calls).

The coordinator orchestrates: Detection → Attribution (for flagged entities) →
ResponseAgent. It can re-invoke AttributionAgent if the ResponseAgent surfaces
new lateral movement candidates, making this a genuine multi-step agentic loop.

Honest scope note: agents are implemented as Python classes calling local
functions — no LLM backend required. The agentic pattern (role separation,
shared context bus, coordinator re-invocation loop) is the architectural
contribution, not the inference engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Agent result and context data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    agent_name: str
    status: str                          # "success" | "no_action" | "error"
    output: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    messages: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    """
    Shared message bus passed between agents. Each agent reads from and
    writes to this context. The coordinator controls the iteration loop.
    """
    # Inputs set by caller
    X_test: Optional[np.ndarray] = None
    entity_ids: Optional[list[str]] = None
    feature_names: Optional[list[str]] = None
    attack_cats: Optional[list[str]] = None   # ground-truth categories (for enrichment)

    # Filled by DetectionAgent
    scores: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    flagged_indices: Optional[list[int]] = None
    flagged_entity_ids: Optional[list[str]] = None
    flagged_scores: Optional[dict[str, float]] = None   # entity_id -> risk_score

    # Filled by AttributionAgent
    attributions: Optional[dict[str, dict]] = None      # entity_id -> enriched attribution
    attack_chains: Optional[dict[str, dict]] = None     # entity_id -> kill-chain chain

    # Filled by ResponseAgent
    soar_records: Optional[list[dict]] = None
    lateral_movement_paths: Optional[list[list]] = None
    new_candidates: Optional[list[str]] = None          # entity IDs surfaced by lateral movement
    bft_results: Optional[dict[str, object]] = None     # entity_id -> ConsensusResult

    # Coordinator bookkeeping
    iteration: int = 0
    trace: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    name: str = "BaseAgent"
    role: str = "undefined"
    capabilities: list[str] = []

    def run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError

    def _timed(self, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Detection Agent
# ---------------------------------------------------------------------------

class DetectionAgent(BaseAgent):
    """
    Scores all entities using HybridAnomalyDetector and identifies flagged
    entities above the anomaly threshold. Also generates per-feature
    anomaly explanations for flagged entities using model.explain_anomaly().
    """
    name = "DetectionAgent"
    role = "Unsupervised anomaly detection and scoring"
    capabilities = ["score_entities", "flag_anomalies", "explain_anomaly"]

    def __init__(self, model):
        self.model = model

    def run(self, context: AgentContext) -> AgentResult:
        t0 = time.perf_counter()
        messages = []

        if context.X_test is None:
            return AgentResult(self.name, "error", messages=["No X_test in context"])

        # Score all entities
        scores = self.model.score(context.X_test)
        predictions = self.model.predict(context.X_test)

        flagged_indices = list(np.where(predictions == 1)[0])
        flagged_entity_ids = []
        flagged_scores = {}

        entity_ids = context.entity_ids or [f"entity_{i}" for i in range(len(scores))]

        for idx in flagged_indices:
            eid = entity_ids[idx]
            flagged_entity_ids.append(eid)
            flagged_scores[eid] = float(scores[idx])

        messages.append(f"Scored {len(scores)} entities, flagged {len(flagged_indices)} anomalies")
        messages.append(f"Score range: [{scores.min():.3f}, {scores.max():.3f}], "
                        f"threshold={self.model.threshold_:.3f}")

        # Generate explanations for top-5 highest-risk flagged entities
        explanations = {}
        if hasattr(self.model, "explain_anomaly") and context.feature_names is not None:
            top_flagged = sorted(flagged_indices, key=lambda i: scores[i], reverse=True)[:5]
            for idx in top_flagged:
                eid = entity_ids[idx]
                explanations[eid] = self.model.explain_anomaly(
                    context.X_test[idx:idx+1], context.feature_names
                )

        # Write to context
        context.scores = scores
        context.predictions = predictions
        context.flagged_indices = flagged_indices
        context.flagged_entity_ids = flagged_entity_ids
        context.flagged_scores = flagged_scores

        duration_ms = (time.perf_counter() - t0) * 1000
        context.trace.append({
            "iteration": context.iteration,
            "agent": self.name,
            "n_flagged": len(flagged_indices),
            "duration_ms": round(duration_ms, 2),
        })

        return AgentResult(
            agent_name=self.name,
            status="success",
            output={
                "n_scored": int(len(scores)),
                "n_flagged": int(len(flagged_indices)),
                "threshold": float(self.model.threshold_),
                "flagged_entity_ids": flagged_entity_ids[:10],  # truncated for log
                "explanations": explanations,
            },
            duration_ms=duration_ms,
            messages=messages,
        )


# ---------------------------------------------------------------------------
# Attribution Agent
# ---------------------------------------------------------------------------

class AttributionAgent(BaseAgent):
    """
    For each flagged entity, retrieves MITRE ATT&CK attribution enriched with
    supporting CVE and CERT-In evidence via the RAGEngine, and traces the
    kill-chain context via the MITREKnowledgeGraph.
    """
    name = "AttributionAgent"
    role = "ATT&CK technique attribution and threat intelligence enrichment"
    capabilities = ["attribute_technique", "retrieve_cve_evidence",
                    "retrieve_advisory_evidence", "trace_kill_chain"]

    def __init__(self, rag_engine, knowledge_graph):
        self.rag = rag_engine
        self.kg = knowledge_graph

    def run(self, context: AgentContext) -> AgentResult:
        t0 = time.perf_counter()
        messages = []

        if not context.flagged_entity_ids:
            return AgentResult(self.name, "no_action",
                               messages=["No flagged entities to attribute"])

        entity_ids = context.entity_ids or []
        attack_cats = context.attack_cats or []

        attributions: dict[str, dict] = {}
        attack_chains: dict[str, dict] = {}

        for eid in context.flagged_entity_ids:
            # Resolve attack category for this entity
            try:
                idx = entity_ids.index(eid)
                cat = attack_cats[idx] if idx < len(attack_cats) else "Unknown"
            except ValueError:
                cat = "Unknown"

            # RAG-enriched attribution
            enriched = self.rag.enrich_attribution(cat)
            attributions[eid] = {
                "attack_cat": cat,
                "technique_doc": enriched.get("technique_doc"),
                "cve_evidence": enriched.get("cve_evidence", []),
                "advisory_evidence": enriched.get("advisory_evidence"),
            }

            # Kill-chain graph traversal
            chain = self.kg.get_attack_chain(cat)
            if "error" not in chain:
                attack_chains[eid] = chain
                messages.append(
                    f"{eid}: {cat} → {chain['technique_id']} [{chain['tactic']}] "
                    f"kill-chain pos {chain['kill_chain_position']}/{chain['kill_chain_total']}"
                )

        context.attributions = attributions
        context.attack_chains = attack_chains

        duration_ms = (time.perf_counter() - t0) * 1000
        context.trace.append({
            "iteration": context.iteration,
            "agent": self.name,
            "n_attributed": len(attributions),
            "duration_ms": round(duration_ms, 2),
        })

        return AgentResult(
            agent_name=self.name,
            status="success",
            output={
                "n_attributed": len(attributions),
                "attack_chains_found": len(attack_chains),
                "sample_attribution": next(iter(attributions.values())) if attributions else None,
            },
            duration_ms=duration_ms,
            messages=messages,
        )


# ---------------------------------------------------------------------------
# Response Agent
# ---------------------------------------------------------------------------

class ResponseAgent(BaseAgent):
    """
    For each flagged entity, selects the appropriate SOAR playbook based on
    the attributed MITRE technique and executes the response decision.
    Also checks for lateral movement candidates and reports new entity IDs
    to the coordinator for potential re-invocation of DetectionAgent.
    """
    name = "ResponseAgent"
    role = "SOAR playbook execution and lateral movement escalation"
    capabilities = ["select_playbook", "execute_response", "detect_lateral_movement",
                    "surface_new_candidates"]

    def __init__(self, orchestrator, graph=None, bft_layer=None):
        self.orchestrator = orchestrator
        self.graph = graph          # optional AttackGraph instance
        self.bft_layer = bft_layer  # optional BFTConsensusLayer instance

    def run(self, context: AgentContext) -> AgentResult:
        t0 = time.perf_counter()
        messages = []

        if not context.flagged_entity_ids:
            return AgentResult(self.name, "no_action",
                               messages=["No flagged entities for response"])

        soar_records = []
        bft_results: dict = {}
        for eid in context.flagged_entity_ids:
            risk_score = context.flagged_scores.get(eid, 0.0) if context.flagged_scores else 0.0

            # Get technique ID from attribution for technique-based playbook selection
            technique_id = None
            if context.attributions and eid in context.attributions:
                td = context.attributions[eid].get("technique_doc")
                if td:
                    # Extract technique ID from doc ID like "MITRE-T1595"
                    technique_id = td["id"].replace("MITRE-", "")

            # BFT consensus vote (if layer is active and we have X_test)
            consensus_result = None
            if self.bft_layer is not None and context.X_test is not None:
                try:
                    entity_ids = context.entity_ids or []
                    idx = entity_ids.index(eid) if eid in entity_ids else -1
                    if 0 <= idx < len(context.X_test):
                        consensus_result = self.bft_layer.vote(
                            eid, context.X_test[idx:idx+1]
                        )
                        bft_results[eid] = consensus_result
                        messages.append(
                            f"BFT [{eid}]: A={consensus_result.votes[0].flagged} "
                            f"B={consensus_result.votes[1].flagged} "
                            f"C={consensus_result.votes[2].flagged} "
                            f"→ {consensus_result.consensus}"
                        )
                except Exception as e:
                    messages.append(f"BFT vote skipped for {eid}: {e}")

            record = self.orchestrator.handle(eid, risk_score, technique_id, consensus_result)
            soar_records.append(record)
            messages.append(
                f"{eid}: {record['decision']} "
                f"[{record.get('automation_coverage', '?')} steps] "
                f"playbook={record.get('playbook_name', 'default')}"
            )

        context.soar_records = soar_records
        context.bft_results = bft_results

        # Lateral movement: check graph for new candidates
        new_candidates = []
        lateral_paths = []
        if self.graph is not None and context.flagged_scores:
            try:
                import networkx as nx
                lm = self.graph.find_lateral_movement(context.flagged_scores)
                lateral_paths = lm.get("paths", [])
                # Surface any nodes in lateral movement paths not already flagged
                flagged_set = set(context.flagged_entity_ids)
                for path in lateral_paths:
                    for node in path:
                        if node not in flagged_set:
                            new_candidates.append(node)
                            flagged_set.add(node)
                if new_candidates:
                    messages.append(
                        f"Lateral movement: {len(lateral_paths)} paths found, "
                        f"{len(new_candidates)} new candidate entities surfaced"
                    )
            except Exception as e:
                messages.append(f"Lateral movement check skipped: {e}")

        context.lateral_movement_paths = lateral_paths
        context.new_candidates = new_candidates

        duration_ms = (time.perf_counter() - t0) * 1000
        context.trace.append({
            "iteration": context.iteration,
            "agent": self.name,
            "n_responses": len(soar_records),
            "new_candidates": len(new_candidates),
            "duration_ms": round(duration_ms, 2),
        })

        return AgentResult(
            agent_name=self.name,
            status="success",
            output={
                "n_responses": len(soar_records),
                "decisions": {r["entity_id"]: r["decision"] for r in soar_records},
                "lateral_movement_paths": len(lateral_paths),
                "new_candidates": new_candidates,
            },
            duration_ms=duration_ms,
            messages=messages,
        )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class CyberResilienceCoordinator:
    """
    Orchestrates the three specialist agents in a loop:

      iteration 0: DetectionAgent → AttributionAgent → ResponseAgent
      if ResponseAgent surfaces new lateral movement candidates:
        iteration 1: AttributionAgent (new candidates only) → ResponseAgent
      (loop ends after max_iterations or when no new candidates)

    This multi-step re-invocation is the core agentic behavior: the
    coordinator acts on new information produced by a downstream agent
    (lateral movement candidates) without requiring human intervention.
    """

    def __init__(self, model, rag_engine, knowledge_graph,
                 orchestrator, graph=None, bft_layer=None, max_iterations: int = 2):
        self.detection_agent = DetectionAgent(model)
        self.attribution_agent = AttributionAgent(rag_engine, knowledge_graph)
        self.response_agent = ResponseAgent(orchestrator, graph, bft_layer)
        self.max_iterations = max_iterations

    def run(self, context: AgentContext) -> dict:
        """
        Execute the full multi-agent pipeline.

        Returns a summary dict with all agent results, the shared context,
        and the coordinator trace.
        """
        all_results: list[AgentResult] = []
        coordinator_log = []

        # --- Iteration 0: full pipeline ---
        context.iteration = 0
        coordinator_log.append({"iteration": 0, "phase": "detection"})
        det_result = self.detection_agent.run(context)
        all_results.append(det_result)

        if det_result.status == "success" and context.flagged_entity_ids:
            coordinator_log.append({"iteration": 0, "phase": "attribution"})
            attr_result = self.attribution_agent.run(context)
            all_results.append(attr_result)

            coordinator_log.append({"iteration": 0, "phase": "response"})
            resp_result = self.response_agent.run(context)
            all_results.append(resp_result)
        else:
            coordinator_log.append({"iteration": 0, "phase": "no_flagged_entities"})

        # --- Subsequent iterations: lateral movement follow-up ---
        for iteration in range(1, self.max_iterations):
            if not context.new_candidates:
                coordinator_log.append({"iteration": iteration, "phase": "terminate_no_new_candidates"})
                break

            context.iteration = iteration
            # Swap in the new candidates as the entity list for re-attribution
            context.flagged_entity_ids = context.new_candidates
            context.flagged_scores = {eid: 0.75 for eid in context.new_candidates}  # escalate band
            context.new_candidates = []

            coordinator_log.append({"iteration": iteration, "phase": "attribution_followup"})
            attr_result = self.attribution_agent.run(context)
            all_results.append(attr_result)

            coordinator_log.append({"iteration": iteration, "phase": "response_followup"})
            resp_result = self.response_agent.run(context)
            all_results.append(resp_result)

        # Persist the audit log
        self.response_agent.orchestrator.audit_log.persist()

        return {
            "iterations_run": context.iteration + 1,
            "agent_results": [
                {
                    "agent": r.agent_name,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "output": r.output,
                    "messages": r.messages,
                }
                for r in all_results
            ],
            "coordinator_log": coordinator_log,
            "context_trace": context.trace,
            "audit_chain_valid": self.response_agent.orchestrator.audit_log.verify_chain(),
        }
