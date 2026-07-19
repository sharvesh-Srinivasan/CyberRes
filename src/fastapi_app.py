import os
import sys
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# Ensure we can import from src
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_raw, prepare_features, make_synthetic_dataset, ATTACK_CATEGORIES
from calendar_features import assign_calendar_phase, add_calendar_features, inject_legitimate_bursts
from temporal_features import add_temporal_features
from model import HybridAnomalyDetector
from evaluate import per_category_metrics, calendar_ablation, fpr_recall_tradeoff
from mitre_rag import attribute, attribution_accuracy
from soar import ResponseOrchestrator, TIER0_WHITELIST
from latency import measure_pipeline_latency
from knowledge_graph import MITREKnowledgeGraph
from rag_engine import RAGEngine
from graph import build_graph
from bft_consensus import BFTConsensusLayer
import numpy as np

app = FastAPI(title="ET // Cyber Command")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Global state to hold pipeline data
PIPELINE_DATA = {}

def _fetch_data_if_missing():
    import urllib.request
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    base = ("https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
            "UNSW-Network_Packet_Classification/")
    for fname in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base + fname, fpath)

def init_pipeline():
    print("Initializing CyberRes Pipeline...")
    use_synthetic = True
    use_temporal = True

    if use_synthetic:
        print("Using synthetic dataset for fast loading.")
        train_df, test_df = make_synthetic_dataset()
    else:
        try:
            _fetch_data_if_missing()
            train_df, test_df = load_raw()
        except Exception:
            print("Real data unavailable — using synthetic fallback.")
            train_df, test_df = make_synthetic_dataset()

    X_train_n, X_train_f, X_test, y_test, cat_test = prepare_features(train_df, test_df)

    if use_temporal:
        X_train_n = add_temporal_features(X_train_n)
        X_train_f = add_temporal_features(X_train_f)
        X_test    = add_temporal_features(X_test)

    feature_names = list(X_test.columns)

    # Baseline
    baseline = HybridAnomalyDetector()
    baseline.fit(X_train_n.values)
    y_pred_base = baseline.predict(X_test.values)
    base_metrics = per_category_metrics(y_test, y_pred_base, cat_test.values)

    # Calendar-conditioned
    tr_phase  = assign_calendar_phase(len(X_train_n))
    te_phase  = assign_calendar_phase(len(X_test), seed=99)
    X_tr_burst = inject_legitimate_bursts(X_train_n.reset_index(drop=True), tr_phase, ["sbytes", "dbytes", "spkts", "dpkts", "sload", "dload", "rate"])
    X_tr_cal   = add_calendar_features(X_tr_burst, tr_phase)
    X_te_cal   = add_calendar_features(X_test, te_phase)

    cal_model = HybridAnomalyDetector()
    cal_model.fit(X_tr_cal.values)
    y_pred_cal = cal_model.predict(X_te_cal.values)
    cal_metrics = per_category_metrics(y_test, y_pred_cal, cat_test.values)

    # Ablation
    norm_mask = y_test == 0
    X_norm = X_test[norm_mask].reset_index(drop=True)
    bph = assign_calendar_phase(len(X_norm), seed=123)
    bph[:] = "exam_period"
    X_burst_cal = add_calendar_features(
        inject_legitimate_bursts(X_norm, bph, ["sbytes", "dbytes", "spkts", "dpkts", "sload", "dload", "rate"]), bph).values
    X_burst_raw = inject_legitimate_bursts(X_norm, bph, ["sbytes", "dbytes", "spkts", "dpkts", "sload", "dload", "rate"]).values
    ablation = calendar_ablation(cal_model, baseline, X_burst_cal, X_burst_raw)

    scores     = cal_model.score(X_te_cal.values)
    entity_ids = [f"10.0.{(i//256)%256}.{i%256}" for i in range(len(scores))]
    flagged_idx = list(np.where(y_pred_cal == 1)[0])

    bft = BFTConsensusLayer(cal_model, tier0_whitelist=TIER0_WHITELIST)

    orchestrator = ResponseOrchestrator()
    soar_records, bft_records = [], {}
    for idx in flagged_idx[:100]:
        eid   = entity_ids[idx]
        score = float(scores[idx])
        cat   = cat_test.values[idx]
        attr  = attribute(cat)
        tech  = attr.get("id") if attr else None
        cr    = bft.vote(eid, X_te_cal.values[idx:idx+1])
        bft_records[eid] = cr
        rec   = orchestrator.handle(eid, score, tech, cr)
        soar_records.append(rec)

    pred_cats = np.where(y_pred_cal == 1, cat_test.values, "Normal")
    attr_acc  = attribution_accuracy(pred_cats.tolist(), cat_test.values.tolist())

    lat_orch = ResponseOrchestrator()
    latency  = measure_pipeline_latency(cal_model, lat_orch, X_te_cal.values, n_sample=200)

    attack_graph = build_graph(entity_ids, scores, cat_test.values.tolist(),
                               tier0_whitelist=TIER0_WHITELIST, top_n=100)
    kg  = MITREKnowledgeGraph()
    rag = RAGEngine()

    global PIPELINE_DATA
    PIPELINE_DATA = dict(
        X_te_cal=X_te_cal, y_test=y_test, y_pred_cal=y_pred_cal,
        cat_test=cat_test, scores=scores, entity_ids=entity_ids,
        feature_names=feature_names, flagged_idx=flagged_idx,
        cal_metrics=cal_metrics, base_metrics=base_metrics, ablation=ablation,
        attr_acc=attr_acc, soar_records=soar_records, bft_records=bft_records,
        attack_graph=attack_graph, kg=kg, rag=rag, latency=latency,
        cal_model=cal_model, orchestrator=orchestrator,
    )
    print("Pipeline initialization complete!")

@app.on_event("startup")
def startup_event():
    import threading
    threading.Thread(target=init_pipeline, daemon=True).start()

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/metrics")

@app.get("/metrics", response_class=HTMLResponse)
async def metrics(request: Request):
    if not PIPELINE_DATA:
        return HTMLResponse("<h1 style='color:white;font-family:monospace;padding:2rem'>SYSTEM INITIALIZING... PLEASE STAND BY.</h1>")
    
    d = PIPELINE_DATA
    cm = d["cal_metrics"]["overall"]
    bm = d["base_metrics"]["overall"]
    
    return templates.TemplateResponse("metrics.html", {
        "request": request,
        "precision": f"{cm.get('precision',0):.4f}",
        "precision_delta": f"{cm.get('precision',0)-bm.get('precision',0):+.4f}",
        "recall": f"{cm.get('recall',0):.4f}",
        "recall_delta": f"{cm.get('recall',0)-bm.get('recall',0):+.4f}",
        "fpr": f"{cm.get('fpr',0):.4f}",
        "fpr_delta": f"{cm.get('fpr',0)-bm.get('fpr',0):+.4f}",
        "burst_fpr_red": f"{d['ablation'].get('relative_fpr_reduction',0)*100:.2f}",
        "cm": cm,
        "n_samples": len(d["y_test"]),
        "latency_ms": f"{d['latency']['end_to_end_latency_ms']['mean']:.0f}",
        "ibm_mtti": f"{d['latency']['baseline_mtti_days_global']}",
    })

@app.get("/entities", response_class=HTMLResponse)
async def entities(request: Request):
    if not PIPELINE_DATA:
        return HTMLResponse("<h1 style='color:white;font-family:monospace;padding:2rem'>SYSTEM INITIALIZING... PLEASE STAND BY.</h1>")
        
    d = PIPELINE_DATA
    rows = []
    soar_lookup = {r["entity_id"]: r for r in d["soar_records"]}
    
    for i, idx in enumerate(d["flagged_idx"][:100]):
        eid = d["entity_ids"][idx]
        score = float(d["scores"][idx])
        cat = d["cat_test"].values[idx]
        soar = soar_lookup.get(eid, {})
        cr = d["bft_records"].get(eid)
        
        rows.append({
            "id": eid,
            "risk_score": round(score, 3),
            "category": cat,
            "decision": soar.get("decision", "—"),
            "consensus": cr.consensus if cr else "—",
            "tier0": eid in TIER0_WHITELIST,
            "votes": cr.votes if cr else []
        })
        
    return templates.TemplateResponse("entities.html", {
        "request": request,
        "entities": rows,
    })

@app.get("/graph", response_class=HTMLResponse)
async def graph(request: Request):
    return templates.TemplateResponse("graph.html", {
        "request": request,
    })

# --- JSON API Endpoints for React Frontend ---

@app.get("/api/status")
async def api_status():
    if not PIPELINE_DATA:
        return JSONResponse({"status": "initializing"})
    return JSONResponse({"status": "ready"})

@app.get("/api/metrics")
async def api_metrics():
    if not PIPELINE_DATA:
        return JSONResponse({"error": "System initializing"}, status_code=503)
    
    d = PIPELINE_DATA
    cm = d["cal_metrics"]["overall"]
    bm = d["base_metrics"]["overall"]
    
    return JSONResponse({
        "precision": cm.get('precision', 0),
        "precision_delta": cm.get('precision', 0) - bm.get('precision', 0),
        "recall": cm.get('recall', 0),
        "recall_delta": cm.get('recall', 0) - bm.get('recall', 0),
        "fpr": cm.get('fpr', 0),
        "fpr_delta": cm.get('fpr', 0) - bm.get('fpr', 0),
        "burst_fpr_red": d['ablation'].get('relative_fpr_reduction', 0) * 100,
        "cm": cm,
        "n_samples": len(d["y_test"]),
        "latency_ms": d['latency']['end_to_end_latency_ms']['mean'],
        "ibm_mtti": d['latency']['baseline_mtti_days_global']
    })

@app.get("/api/entities")
async def api_entities():
    if not PIPELINE_DATA:
        return JSONResponse({"error": "System initializing"}, status_code=503)
        
    d = PIPELINE_DATA
    rows = []
    soar_lookup = {r["entity_id"]: r for r in d["soar_records"]}
    
    for idx in d["flagged_idx"][:100]:
        eid = d["entity_ids"][idx]
        score = float(d["scores"][idx])
        cat = str(d["cat_test"].values[idx])
        soar = soar_lookup.get(eid, {})
        cr = d["bft_records"].get(eid)
        
        rows.append({
            "id": eid,
            "risk_score": score,
            "category": cat,
            "decision": soar.get("decision", "—"),
            "consensus": cr.consensus if cr else "—",
            "tier0": eid in TIER0_WHITELIST,
            "votes": cr.votes if cr else []
        })
        
    return JSONResponse({"entities": rows})

@app.get("/api/graph")
async def api_graph():
    if not PIPELINE_DATA:
        return JSONResponse({"error": "System initializing"}, status_code=503)
    # The attack_graph is a NetworkX graph. We need to serialize it to JSON.
    from networkx.readwrite import json_graph
    data = json_graph.node_link_data(PIPELINE_DATA["attack_graph"])
    return JSONResponse(data)

@app.get("/api/audit")
async def api_audit():
    if not PIPELINE_DATA:
        return JSONResponse({"error": "System initializing"}, status_code=503)
    
    soar_records = PIPELINE_DATA["soar_records"]
    return JSONResponse({"audit_logs": soar_records})

from pydantic import BaseModel

class OverrideRequest(BaseModel):
    entity_id: str
    action: str

@app.post("/api/override")
async def api_override(req: OverrideRequest):
    if not PIPELINE_DATA:
        return JSONResponse({"error": "System initializing"}, status_code=503)
        
    soar_records = PIPELINE_DATA["soar_records"]
    for record in soar_records:
        if record["entity_id"] == req.entity_id:
            record["decision"] = req.action
            record["override_applied"] = True
            return JSONResponse({"status": "success", "record": record})
            
    return JSONResponse({"error": "Entity not found"}, status_code=404)


