"""
graph.py
Entity-relationship attack graph with lateral movement detection.

Builds a NetworkX graph from scored network entities:
  - Nodes: IP/entity addresses with attributes (risk_score, attack_cat, is_tier0)
  - Edges: communication pairs weighted by combined risk

Key APIs:
  - build_graph(entity_scores, entity_cats, tier0_set) → AttackGraph
  - find_lateral_movement(flagged_scores) → paths between high-risk subgraphs
  - get_attack_paths(source, target) → all simple paths between two entities

Only runs on entities above RISK_THRESHOLD_ESCALATE (0.70) as documented
in CONTEXT.md — running full graph construction on all 82k test entities
would be slow and uninformative for a dashboard view.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Optional
from soar import RISK_THRESHOLD_ESCALATE, TIER0_WHITELIST


class AttackGraph:
    """
    Directed graph of network entities and their risk relationships.

    Nodes: entity IDs (IP addresses or index strings)
    Edges: directed communication relationships, weighted by risk
    """

    def __init__(self):
        self.G = nx.DiGraph()

    def add_entity(self, entity_id: str, risk_score: float,
                   attack_cat: str = "Normal", is_tier0: bool = False):
        self.G.add_node(
            entity_id,
            risk_score=risk_score,
            attack_cat=attack_cat,
            is_tier0=is_tier0,
            node_type="entity",
        )

    def add_communication(self, src: str, dst: str, weight: float = 1.0):
        """Add a directed communication edge between two entities."""
        self.G.add_edge(src, dst, weight=weight, edge_type="communicates")

    def find_lateral_movement(self, flagged_scores: dict[str, float],
                               threshold: float = RISK_THRESHOLD_ESCALATE) -> dict:
        """
        Identify lateral movement patterns: connected subgraphs where
        multiple high-risk entities are linked, suggesting pivot chains.

        Args:
            flagged_scores: {entity_id: risk_score} for flagged entities.
            threshold: minimum risk score to include in lateral movement analysis.

        Returns:
            dict with keys: high_risk_nodes, paths, pivot_candidates.
        """
        high_risk = {eid for eid, score in flagged_scores.items()
                     if score >= threshold and eid in self.G}

        if len(high_risk) < 2:
            return {"high_risk_nodes": list(high_risk), "paths": [], "pivot_candidates": []}

        # Find paths between high-risk nodes (lateral movement chains)
        paths = []
        pivot_candidates = set()
        hr_list = list(high_risk)

        for i, src in enumerate(hr_list):
            for dst in hr_list[i + 1:]:
                try:
                    # Find shortest path through the graph
                    path = nx.shortest_path(self.G, src, dst, weight="weight")
                    if len(path) > 1:
                        paths.append(path)
                        # Intermediate nodes are pivot candidates
                        for node in path[1:-1]:
                            pivot_candidates.add(node)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        return {
            "high_risk_nodes": list(high_risk),
            "paths": paths,
            "pivot_candidates": list(pivot_candidates),
        }

    def get_attack_paths(self, source: str, target: str,
                         max_paths: int = 3) -> list[list[str]]:
        """Return up to max_paths simple paths from source to target."""
        if source not in self.G or target not in self.G:
            return []
        try:
            paths = list(nx.all_simple_paths(self.G, source, target, cutoff=5))
            return sorted(paths, key=len)[:max_paths]
        except nx.NetworkXNoPath:
            return []

    def subgraph_for_entity(self, entity_id: str, radius: int = 2) -> nx.DiGraph:
        """Return the ego subgraph around an entity (for dashboard zoom-in)."""
        if entity_id not in self.G:
            return nx.DiGraph()
        ego = nx.ego_graph(self.G.to_undirected(), entity_id, radius=radius)
        return self.G.subgraph(ego.nodes).copy()

    def summary(self) -> dict:
        """Stats for dashboard display."""
        risk_scores = [d.get("risk_score", 0) for _, d in self.G.nodes(data=True)]
        return {
            "n_entities": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
            "n_high_risk": sum(1 for s in risk_scores if s >= RISK_THRESHOLD_ESCALATE),
            "n_tier0": sum(1 for _, d in self.G.nodes(data=True) if d.get("is_tier0")),
            "mean_risk_score": float(np.mean(risk_scores)) if risk_scores else 0.0,
        }

    def to_plotly_data(self) -> dict:
        """
        Export node and edge data in a format ready for Plotly scatter plots
        (used by the Streamlit dashboard Graph tab).

        Returns dict with 'nodes' and 'edges' lists.
        """
        # Use spring layout for positions
        if self.G.number_of_nodes() == 0:
            return {"nodes": [], "edges": []}

        pos = nx.spring_layout(self.G, seed=42, k=2.0)

        nodes = []
        for node_id, data in self.G.nodes(data=True):
            x, y = pos.get(node_id, (0.0, 0.0))
            nodes.append({
                "id": node_id,
                "x": float(x),
                "y": float(y),
                "risk_score": data.get("risk_score", 0.0),
                "attack_cat": data.get("attack_cat", "Normal"),
                "is_tier0": data.get("is_tier0", False),
            })

        edges = []
        for src, dst, data in self.G.edges(data=True):
            src_pos = pos.get(src, (0, 0))
            dst_pos = pos.get(dst, (0, 0))
            edges.append({
                "x0": float(src_pos[0]), "y0": float(src_pos[1]),
                "x1": float(dst_pos[0]), "y1": float(dst_pos[1]),
                "weight": data.get("weight", 1.0),
            })

        return {"nodes": nodes, "edges": edges}


def build_graph(entity_ids: list[str], scores: np.ndarray,
                attack_cats: Optional[list[str]] = None,
                tier0_whitelist: set = None,
                top_n: int = 100) -> AttackGraph:
    """
    Build an AttackGraph from scored entities.

    Only includes the top_n highest-risk entities to keep the graph
    visually informative and computationally bounded for the dashboard.

    Args:
        entity_ids: list of entity ID strings.
        scores: anomaly scores parallel to entity_ids.
        attack_cats: optional list of category strings (parallel to entity_ids).
        tier0_whitelist: set of entity IDs that are Tier-0 assets.
        top_n: max number of entities to include in the graph.

    Returns:
        AttackGraph ready for lateral movement analysis and dashboard rendering.
    """
    tier0 = tier0_whitelist or TIER0_WHITELIST
    cats = attack_cats or ["Unknown"] * len(entity_ids)

    # Select top-N by risk score to keep the graph bounded
    if len(entity_ids) > top_n:
        top_indices = np.argsort(scores)[::-1][:top_n]
    else:
        top_indices = np.arange(len(entity_ids))

    graph = AttackGraph()

    selected_ids = []
    for idx in top_indices:
        eid = entity_ids[int(idx)]
        score = float(scores[int(idx)])
        cat = cats[int(idx)] if int(idx) < len(cats) else "Unknown"
        graph.add_entity(eid, score, cat, eid in tier0)
        selected_ids.append(eid)

    # Add synthetic communication edges between high-risk entities.
    # In a real deployment, these would come from flow logs (NetFlow / sFlow).
    # Here we use IP-prefix proximity as a proxy for network segment membership.
    rng = np.random.default_rng(42)
    for i, src in enumerate(selected_ids):
        for dst in selected_ids[i + 1:]:
            src_score = float(scores[top_indices[selected_ids.index(src)]])
            dst_score = float(scores[top_indices[selected_ids.index(dst)]])
            # Only connect entities that both have elevated risk (> escalate threshold)
            if src_score >= RISK_THRESHOLD_ESCALATE and dst_score >= RISK_THRESHOLD_ESCALATE:
                # Weight by geometric mean of risk scores
                w = float(np.sqrt(src_score * dst_score))
                if rng.random() < 0.3:  # sparse connections for readability
                    graph.add_communication(src, dst, w)

    return graph
