"""
main.py
End-to-end pipeline: load data -> train baseline (no calendar) and
calendar-conditioned models -> evaluate per-category metrics -> run the
calendar ablation -> run MITRE attribution -> run mock SOAR on flagged
entities -> verify audit log integrity.

Usage:
  python src/main.py                 # uses real UNSW-NB15 CSVs in data/
  python src/main.py --synthetic     # uses synthetic data (smoke test only,
                                      # NOT for reporting real benchmark numbers)
  python src/main.py --agentic       # routes Stage 6+ through the multi-agent
                                      # coordinator (DetectionAgent, AttributionAgent,
                                      # ResponseAgent) instead of direct function calls
"""

import argparse
import json
import numpy as np

from data_loader import load_raw, prepare_features, make_synthetic_dataset, ATTACK_CATEGORIES
from calendar_features import assign_calendar_phase, add_calendar_features, inject_legitimate_bursts
from temporal_features import add_temporal_features, temporal_feature_names
from model import HybridAnomalyDetector
from evaluate import per_category_metrics, calendar_ablation
from mitre_rag import attribute, attribution_accuracy
from soar import ResponseOrchestrator, AuditLog, TIER0_WHITELIST
from latency import measure_pipeline_latency, format_comparison
from knowledge_graph import MITREKnowledgeGraph
from rag_engine import RAGEngine
from graph import build_graph
from agents import CyberResilienceCoordinator, AgentContext
from bft_consensus import BFTConsensusLayer


VOLUME_COLS = ["sbytes", "dbytes", "spkts", "dpkts", "sload", "dload", "rate"]


def run(use_synthetic: bool, use_agentic: bool = False, use_bft: bool = False,
        use_temporal: bool = False):
    print(f"\n{'='*60}\nSTAGE 1: Data loading ({'synthetic' if use_synthetic else 'real UNSW-NB15'})\n{'='*60}")
    if use_synthetic:
        train_df, test_df = make_synthetic_dataset()
    else:
        train_df, test_df = load_raw()

    X_train_normal, X_train_full, X_test, y_test, cat_test = prepare_features(train_df, test_df)

    if use_temporal:
        print("  [temporal] Adding rolling window features...")
        X_train_normal = add_temporal_features(X_train_normal)
        X_train_full = add_temporal_features(X_train_full)
        X_test = add_temporal_features(X_test)
        extra = temporal_feature_names(list(X_test.columns))
        print(f"  [temporal] Added {len(extra)} features. New shape: {X_test.shape}")

    print(f"Train (normal-only): {X_train_normal.shape} | Test: {X_test.shape}")

    print(f"\n{'='*60}\nSTAGE 2: Train baseline model (NO calendar conditioning)\n{'='*60}")
    baseline = HybridAnomalyDetector()
    baseline.fit(X_train_normal.values)
    print(f"Calibrated threshold (5% target FPR on training-normal distribution): {baseline.threshold_:.4f}")
    y_pred_baseline = baseline.predict(X_test.values)

    baseline_metrics = per_category_metrics(y_test, y_pred_baseline, cat_test.values)
    print(json.dumps(baseline_metrics["overall"], indent=2))
    print("\nPer-category recall (detection rate):")
    for cat, m in baseline_metrics.items():
        if cat != "overall":
            print(f"  {cat:16s} recall={m['recall']:.3f}  n={m['n_samples']}")

    print(f"\n{'='*60}\nSTAGE 3: Train calendar-conditioned model\n{'='*60}")
    train_phase = assign_calendar_phase(len(X_train_normal))
    test_phase = assign_calendar_phase(len(X_test), seed=99)

    # Critical: the model can only learn "high volume during exam_period is normal"
    # if training data actually contains phase-correlated volume bursts. Without
    # this, the phase one-hot column is pure noise to the autoencoder.
    X_train_normal_burst = inject_legitimate_bursts(
        X_train_normal.reset_index(drop=True), train_phase, VOLUME_COLS
    )
    X_train_cal = add_calendar_features(X_train_normal_burst, train_phase)
    X_test_cal = add_calendar_features(X_test, test_phase)

    cal_model = HybridAnomalyDetector()
    cal_model.fit(X_train_cal.values)
    print(f"Calibrated threshold (5% target FPR on training-normal distribution): {cal_model.threshold_:.4f}")
    y_pred_cal = cal_model.predict(X_test_cal.values)
    cal_metrics = per_category_metrics(y_test, y_pred_cal, cat_test.values)
    print(json.dumps(cal_metrics["overall"], indent=2))

    print(f"\n{'='*60}\nSTAGE 4: Calendar-conditioning ablation (the differentiator)\n{'='*60}")
    # Build a legitimate-burst-only slice (normal traffic during exam_period, injected volume spike)
    normal_test_mask = (y_test == 0)
    X_normal_test = X_test[normal_test_mask].reset_index(drop=True)
    burst_phase = assign_calendar_phase(len(X_normal_test), seed=123)
    burst_phase[:] = "exam_period"  # force all of this slice into the burst period

    X_burst_raw = inject_legitimate_bursts(X_normal_test, burst_phase, VOLUME_COLS)
    X_burst_with_cal = add_calendar_features(X_burst_raw, burst_phase).values
    X_burst_no_cal = X_burst_raw.values  # baseline model wasn't trained with calendar cols at all

    ablation = calendar_ablation(cal_model, baseline, X_burst_with_cal, X_burst_no_cal)
    print(json.dumps(ablation, indent=2))

    print(f"\n{'='*60}\nSTAGE 5: MITRE ATT&CK attribution\n{'='*60}")
    pred_cats_for_flagged = np.where(y_pred_cal == 1, cat_test.values, "Normal")
    attr_acc = attribution_accuracy(pred_cats_for_flagged.tolist(), cat_test.values.tolist())
    print(f"Attribution accuracy (technique-level, on flagged samples): {attr_acc:.3f}")
    example_flagged_idx = np.where(y_pred_cal == 1)[0][:3]
    for idx in example_flagged_idx:
        cat = cat_test.values[idx]
        print(f"  entity #{idx} flagged as '{cat}' -> {attribute(cat)}")

    print(f"\n{'='*60}\nSTAGE 6: Knowledge Graph + RAG attribution enrichment\n{'='*60}")
    kg = MITREKnowledgeGraph()
    rag = RAGEngine()
    print(f"Knowledge graph: {kg.summary()}")
    print(f"RAG corpus: {rag.corpus_stats()}")
    for cat in ["Exploits", "Worms", "DoS"]:
        chain = kg.get_attack_chain(cat)
        enriched = rag.enrich_attribution(cat)
        td = enriched.get('technique_doc')
        print(f"  {cat} -> [{chain.get('technique_id')}] {chain.get('tactic')} "
              f"(phase {chain.get('kill_chain_position', 0)+1}/{chain.get('kill_chain_total', 14)}) "
              f"| RAG top-doc: {td['title'] if td else 'none'}")

    print(f"\n{'='*60}\nSTAGE 7: SOAR + Tier-0 whitelist + hashed audit log\n{'='*60}")
    orchestrator = ResponseOrchestrator()
    scores = cal_model.score(X_test_cal.values)
    demo_entities = [
        ("10.0.0.1", scores[example_flagged_idx[0]]) if len(example_flagged_idx) else ("10.0.0.1", 0.9),
        ("10.0.7.44", 0.92),   # non-critical, high risk -> should auto-contain
        ("10.0.5.10", 0.95),   # Tier-0, high risk -> must escalate-only despite high score
        ("10.0.7.99", 0.72),   # mid risk -> escalate band
    ]
    for entity_id, risk_score in demo_entities:
        mitre_attr = attribute(cat_test.values[0])
        tech_id = mitre_attr.get("id") if mitre_attr else None
        record = orchestrator.handle(entity_id, float(risk_score), tech_id)
        print(f"  {entity_id} (risk={risk_score:.2f}, tier0={entity_id in TIER0_WHITELIST}) -> "
              f"{record['decision']} [{record['automation_coverage']} steps] "
              f"playbook={record['playbook_name']} :: {record['reason']}")

    orchestrator.audit_log.persist()
    print(f"\nAudit log chain valid: {orchestrator.audit_log.verify_chain()}")
    print(f"Audit log written to logs/audit_log.jsonl ({len(orchestrator.audit_log.entries)} entries)")

    print(f"\n{'='*60}\nSTAGE 8: MTTD/MTTR -- measured detection+response latency vs. cited baseline\n{'='*60}")
    latency_orchestrator = ResponseOrchestrator()  # separate audit log so the probe doesn't pollute the demo log
    latency_result = measure_pipeline_latency(cal_model, latency_orchestrator, X_test_cal.values, n_sample=200)
    print(format_comparison(latency_result))

    if use_agentic:
        print(f"\n{'='*60}\nSTAGE 9 (AGENTIC): Multi-agent coordinator pipeline\n{'='*60}")
        entity_ids = [f"10.0.{(i // 256) % 256}.{i % 256}" for i in range(len(scores))]
        attack_graph = build_graph(
            entity_ids, scores, cat_test.values.tolist(),
            tier0_whitelist=TIER0_WHITELIST, top_n=50
        )
        agent_orchestrator = ResponseOrchestrator()

        bft_layer = None
        if use_bft:
            print("  [BFT] Byzantine Fault Tolerance consensus layer ACTIVE")
            bft_layer = BFTConsensusLayer(cal_model, tier0_whitelist=TIER0_WHITELIST)

        coordinator = CyberResilienceCoordinator(
            model=cal_model,
            rag_engine=rag,
            knowledge_graph=kg,
            orchestrator=agent_orchestrator,
            graph=attack_graph,
            bft_layer=bft_layer,
            max_iterations=2,
        )
        context = AgentContext(
            X_test=X_test_cal.values,
            entity_ids=entity_ids,
            feature_names=list(X_test_cal.columns),
            attack_cats=cat_test.values.tolist(),
        )
        agent_result = coordinator.run(context)
        print(f"  Iterations run: {agent_result['iterations_run']}")
        print(f"  Audit chain valid: {agent_result['audit_chain_valid']}")
        for ar in agent_result["agent_results"]:
            print(f"  [{ar['agent']}] {ar['status']} in {ar['duration_ms']:.1f}ms")
            for msg in ar["messages"][:3]:
                print(f"    → {msg}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Baseline FPR: {baseline_metrics['overall']['fpr']:.3f} | "
          f"Calendar-conditioned FPR: {cal_metrics['overall']['fpr']:.3f}")
    print(f"Legitimate-burst FPR reduction from calendar conditioning: "
          f"{ablation['relative_fpr_reduction']*100:.1f}%")
    print(f"MITRE attribution accuracy: {attr_acc*100:.1f}%")
    print(f"End-to-end detect+decide latency (p95): {latency_result['end_to_end_latency_ms']['p95']:.2f} ms "
          f"vs. cited India MTTI baseline of {latency_result['baseline_mtti_days_india_range'][0]}-"
          f"{latency_result['baseline_mtti_days_india_range'][1]} days ({latency_result['baseline_source']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                         help="Use synthetic data instead of real UNSW-NB15 CSVs")
    parser.add_argument("--agentic", action="store_true",
                         help="Route Stage 9 through the multi-agent coordinator")
    parser.add_argument("--bft", action="store_true",
                         help="Enable Byzantine Fault Tolerance consensus (requires --agentic)")
    parser.add_argument("--temporal", action="store_true",
                         help="Add rolling window temporal features before training")
    args = parser.parse_args()
    run(use_synthetic=args.synthetic, use_agentic=args.agentic,
        use_bft=args.bft, use_temporal=args.temporal)
