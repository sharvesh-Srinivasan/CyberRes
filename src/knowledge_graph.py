"""
knowledge_graph.py
MITRE ATT&CK Knowledge Graph over the 9 UNSW-NB15 attack categories.

Builds a directed NetworkX graph encoding:
  - 14 ATT&CK tactic nodes (ordered by kill-chain phase)
  - 9 technique nodes (one per UNSW-NB15 category, mapped from mitre_rag.py)
  - 9 category nodes (the UNSW-NB15 attack_cat labels)
  - Edges: category --[maps_to]--> technique --[belongs_to]--> tactic
  - Kill-chain ordering: tactic --[precedes]--> tactic (ATT&CK sequence)
  - Lateral movement links: technique --[enables]--> technique across phases

Honest scope note: the technique nodes and their tactic assignments come from
the same curated mapping as mitre_rag.py -- this adds GRAPH STRUCTURE over
that flat dict, enabling path queries (kill-chain traversal, related-technique
lookup) that the dict cannot provide.
"""

import networkx as nx
from typing import Optional

# ---------------------------------------------------------------------------
# ATT&CK kill-chain tactic order (from reconnaissance to impact)
# ---------------------------------------------------------------------------
TACTIC_ORDER = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]

# ---------------------------------------------------------------------------
# UNSW-NB15 category → technique → tactic (same mapping as mitre_rag.py,
# extended with campaign / sub-technique context for graph enrichment)
# ---------------------------------------------------------------------------
CATEGORY_TECHNIQUE_MAP = {
    "Fuzzers": {
        "technique_id": "T1595",
        "technique_name": "Active Scanning",
        "tactic": "Reconnaissance",
        "description": "Adversaries scan victim infrastructure to gather information prior to exploitation.",
        "related_techniques": ["T1592", "T1190"],
    },
    "Analysis": {
        "technique_id": "T1592",
        "technique_name": "Gather Victim Host Information",
        "tactic": "Reconnaissance",
        "description": "Adversaries gather information about the victim's hosts for target selection.",
        "related_techniques": ["T1595", "T1190"],
    },
    "Backdoor": {
        "technique_id": "T1547",
        "technique_name": "Boot or Logon Autostart Execution",
        "tactic": "Persistence",
        "description": "Adversaries achieve persistence by placing malicious code in autostart locations.",
        "related_techniques": ["T1059", "T1210"],
    },
    "DoS": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact",
        "description": "Adversaries perform denial of service attacks to degrade or block availability.",
        "related_techniques": ["T1071"],
    },
    "Exploits": {
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
        "description": "Adversaries exploit weaknesses in internet-facing applications to gain access.",
        "related_techniques": ["T1595", "T1059", "T1547"],
    },
    "Generic": {
        "technique_id": "T1071",
        "technique_name": "Application Layer Protocol",
        "tactic": "Command and Control",
        "description": "Adversaries communicate with compromised systems using standard application layer protocols.",
        "related_techniques": ["T1498", "T1547"],
    },
    "Reconnaissance": {
        "technique_id": "T1595",
        "technique_name": "Active Scanning",
        "tactic": "Reconnaissance",
        "description": "Adversaries actively scan victim network infrastructure to gather information.",
        "related_techniques": ["T1592", "T1190"],
    },
    "Shellcode": {
        "technique_id": "T1059",
        "technique_name": "Command and Scripting Interpreter",
        "tactic": "Execution",
        "description": "Adversaries abuse command and script interpreters to execute arbitrary commands.",
        "related_techniques": ["T1547", "T1210"],
    },
    "Worms": {
        "technique_id": "T1210",
        "technique_name": "Exploitation of Remote Services",
        "tactic": "Lateral Movement",
        "description": "Adversaries exploit remote services to move laterally through the network.",
        "related_techniques": ["T1059", "T1071"],
    },
}


class MITREKnowledgeGraph:
    """
    Directed graph encoding MITRE ATT&CK structure over the UNSW-NB15
    attack categories. Use get_attack_chain() to traverse the kill-chain
    from a detection category, and get_related_techniques() to surface
    adjacent threat intelligence.
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self._build()

    def _build(self):
        G = self.G

        # --- Tactic nodes ---
        for i, tactic in enumerate(TACTIC_ORDER):
            G.add_node(tactic, node_type="tactic", kill_chain_position=i)

        # --- Kill-chain ordering edges (tactic → tactic) ---
        for i in range(len(TACTIC_ORDER) - 1):
            G.add_edge(TACTIC_ORDER[i], TACTIC_ORDER[i + 1],
                       edge_type="precedes", weight=1.0)

        # --- Technique and category nodes ---
        seen_techniques = {}
        for category, meta in CATEGORY_TECHNIQUE_MAP.items():
            tid = meta["technique_id"]
            tname = meta["technique_name"]
            tactic = meta["tactic"]

            # Technique node (deduplicated: Fuzzers and Reconnaissance both map T1595)
            if tid not in seen_techniques:
                G.add_node(tid, node_type="technique",
                           name=tname, tactic=tactic,
                           description=meta["description"])
                seen_techniques[tid] = tactic
                # technique → tactic
                G.add_edge(tid, tactic, edge_type="belongs_to", weight=1.0)

            # Category node
            G.add_node(category, node_type="category",
                       technique_id=tid, technique_name=tname)
            # category → technique
            G.add_edge(category, tid, edge_type="maps_to", weight=1.0)

            # Inter-technique "enables" edges (cross-tactic chaining)
            for related_tid in meta.get("related_techniques", []):
                if related_tid != tid:
                    # Add technique node stub if not yet seen
                    if not G.has_node(related_tid):
                        G.add_node(related_tid, node_type="technique",
                                   name=related_tid, tactic="Unknown")
                    G.add_edge(tid, related_tid, edge_type="enables", weight=0.5)

    # -----------------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------------

    def get_attack_chain(self, category: str) -> dict:
        """
        Return the full ATT&CK kill-chain context for a detected category:
        category → technique → tactic, plus the full kill-chain position
        and which subsequent tactics are reachable from this entry point.

        Returns a dict with keys: category, technique_id, technique_name,
        tactic, kill_chain_position, subsequent_tactics, related_technique_ids.
        """
        if category not in CATEGORY_TECHNIQUE_MAP:
            return {"category": category, "error": "Unknown category"}

        meta = CATEGORY_TECHNIQUE_MAP[category]
        tid = meta["technique_id"]
        tactic = meta["tactic"]
        position = TACTIC_ORDER.index(tactic) if tactic in TACTIC_ORDER else -1

        # Subsequent tactics reachable from this tactic in the kill-chain
        subsequent = TACTIC_ORDER[position + 1:] if position >= 0 else []

        # Related technique IDs from "enables" edges
        related = [
            v for u, v, d in self.G.out_edges(tid, data=True)
            if d.get("edge_type") == "enables"
        ]

        return {
            "category": category,
            "technique_id": tid,
            "technique_name": meta["technique_name"],
            "tactic": tactic,
            "kill_chain_position": position,
            "kill_chain_total": len(TACTIC_ORDER),
            "subsequent_tactics": subsequent[:4],  # next 4 phases only for brevity
            "related_technique_ids": related,
            "description": meta["description"],
        }

    def get_related_techniques(self, technique_id: str) -> list[dict]:
        """
        Return techniques reachable from technique_id via 'enables' edges,
        with their names and tactics. Used by AttributionAgent for surface-
        level threat intelligence enrichment.
        """
        results = []
        if not self.G.has_node(technique_id):
            return results
        for _, v, data in self.G.out_edges(technique_id, data=True):
            if data.get("edge_type") == "enables":
                node_data = self.G.nodes[v]
                results.append({
                    "technique_id": v,
                    "technique_name": node_data.get("name", v),
                    "tactic": node_data.get("tactic", "Unknown"),
                })
        return results

    def get_tactic_neighbors(self, tactic: str) -> dict:
        """Return the predecessor and successor tactics for a given tactic."""
        if tactic not in self.G:
            return {}
        predecessors = [
            u for u, v, d in self.G.in_edges(tactic, data=True)
            if d.get("edge_type") == "precedes"
        ]
        successors = [
            v for u, v, d in self.G.out_edges(tactic, data=True)
            if d.get("edge_type") == "precedes"
        ]
        return {"tactic": tactic, "preceded_by": predecessors, "leads_to": successors}

    def shortest_attack_path(self, from_category: str, to_category: str) -> Optional[list]:
        """
        Return the shortest path between two attack categories through
        the technique/tactic graph. Returns None if no path exists.
        """
        try:
            return nx.shortest_path(self.G, from_category, to_category)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def summary(self) -> dict:
        """Return graph statistics for dashboard display."""
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "tactic_nodes": sum(1 for _, d in self.G.nodes(data=True) if d.get("node_type") == "tactic"),
            "technique_nodes": sum(1 for _, d in self.G.nodes(data=True) if d.get("node_type") == "technique"),
            "category_nodes": sum(1 for _, d in self.G.nodes(data=True) if d.get("node_type") == "category"),
        }
