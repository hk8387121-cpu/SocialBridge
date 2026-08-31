"""
app.py — SocialBridge Flask Application

Social Network Analytics & Influence Optimisation
DAA Capstone Project
"""
from __future__ import annotations

import json
import logging
import os
import time
from functools import wraps
from typing import Any, Dict

import networkx as nx
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from config import Config
from services.data_processor import process_dataset, allowed_file
from services.graph_builder import build_graph, get_largest_component, sample_graph
from services.graph_statistics import compute_basic_stats, compute_advanced_stats, compute_centrality
from services.community_detection import run_greedy_modularity, run_louvain, run_girvan_newman
from services.community_metrics import compute_community_details
from services.community_graph import build_community_graph
from services.influence_model import simulate_ic, simulate_lt
from services.greedy_optimizer import run_greedy
from services.celf_optimizer import run_celf
from services.comparison_engine import run_all_baselines, run_scalability_analysis
from services.visualization import build_graph_viz, build_community_overview_viz, build_community_detail_viz
from services.sample_generator import generate_network
from services.report_generator import save_experiment, load_experiments, generate_conclusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config.from_object(Config)
for key in ("UPLOAD_FOLDER", "DATA_RAW_DIR", "DATA_PROC_DIR", "RESULTS_DIR"):
    os.makedirs(app.config[key], exist_ok=True)
STATE: Dict[str, Any] = {"dataset_name": None, "preprocessing": None, "graph": None, "graph_type": "undirected", "basic_stats": None, "advanced_stats": None, "centrality": None, "community_results": {}, "active_comm_alg": None, "community_graph": None, "community_details": None, "influence_results": {}, "active_influence": None, "scalability": None, "experiment_ts": None}

def ok(data: Dict = None, **kwargs) -> Any:
    payload = {"status": "ok"}
    if data: payload.update(data)
    payload.update(kwargs)
    return jsonify(payload)

def err(message: str, code: int = 400) -> Any:
    return jsonify({"status": "error", "message": message}), code

def graph_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if STATE["graph"] is None: return err("No graph loaded. Upload and process a dataset first.")
        return fn(*args, **kwargs)
    return wrapper

def communities_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not STATE["community_results"]: return err("No community detection results. Run community detection first.")
        return fn(*args, **kwargs)
    return wrapper

def _get_analysis_graph(mode: str) -> nx.Graph:
    G = STATE["graph"]
    if mode == "lcc": return get_largest_component(G)
    if mode == "sampled": return sample_graph(G, max_nodes=app.config["GN_MAX_NODES"])[0]
    return G

def _get_node_info(G: nx.Graph, node: int) -> Dict:
    cent = STATE.get("centrality") or {}; alg = STATE.get("active_comm_alg")
    node_comm = STATE["community_results"].get(alg, {}).get("node_community", {}) if alg else {}
    return {"node": node, "degree": G.degree(node), "community": node_comm.get(node, "N/A"), "degree_cent": next((e["value"] for e in cent.get("degree", []) if e["node"] == node), "N/A"), "betweenness": next((e["value"] for e in cent.get("betweenness", []) if e["node"] == node), "N/A"), "closeness": next((e["value"] for e in cent.get("closeness", []) if e["node"] == node), "N/A")}

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email, password = request.form.get("email", "").strip(), request.form.get("password", "")
        if email and password:
            session.update(user_email=email, logged_in=True); return redirect(url_for("index"))
        from flask import flash; flash("Invalid email or password. Please try again.", "error"); return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    from flask import flash
    import datetime
    if request.method == "POST":
        email, password, confirm, terms = request.form.get("email", "").strip(), request.form.get("password", ""), request.form.get("confirm_password", ""), request.form.get("terms")
        if not email or not password: flash("Email and password are required.", "error"); return redirect(url_for("register"))
        if password != confirm: flash("Passwords do not match. Please try again.", "error"); return redirect(url_for("register"))
        if not terms: flash("You must accept the Terms of Service to register.", "error"); return redirect(url_for("register"))
        session["user_email"], session["logged_in"] = email, True
        session["user_profile"] = {"full_name": request.form.get("full_name", "").strip(), "email": email, "phone": request.form.get("phone", "").strip(), "job_title": request.form.get("job_title", "").strip(), "department": request.form.get("department", "").strip(), "brand_name": request.form.get("brand_name", "").strip(), "website": request.form.get("website", "").strip(), "linkedin": "", "industry": "", "org_size": "", "org_description": "", "bio": "", "member_since": datetime.date.today().strftime("%d %b %Y")}
        flash("Account created! Welcome to SocialBridge.", "success"); return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    from flask import flash
    if request.method == "POST": flash("If that email exists, a reset link has been sent.", "success")
    return redirect(url_for("login"))

@app.route("/auth/google")
def login_google():
    import datetime
    session["user_email"], session["logged_in"] = "google-user@socialbridge.app", True
    session.setdefault("user_profile", {"full_name": "Google User", "email": "google-user@socialbridge.app", "phone": "", "job_title": "Analyst", "department": "", "brand_name": "", "website": "", "linkedin": "", "industry": "", "org_size": "", "org_description": "", "bio": "", "member_since": datetime.date.today().strftime("%d %b %Y")})
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

@app.route("/profile")
def profile_page():
    profile = session.get("user_profile", {"full_name": "", "email": session.get("user_email", ""), "phone": "", "job_title": "", "department": "", "brand_name": "", "website": "", "linkedin": "", "industry": "", "org_size": "", "org_description": "", "bio": "", "member_since": "—"})
    return render_template("profile.html", profile=profile)

@app.route("/profile/update", methods=["POST"])
def profile_update():
    from flask import flash
    section, profile = request.form.get("section", "personal"), session.get("user_profile", {})
    if section == "personal":
        for k in ("full_name", "phone", "job_title", "department", "bio"): profile[k] = request.form.get(k, "").strip()
    elif section == "organisation":
        for k in ("brand_name", "website", "linkedin", "industry", "org_size", "org_description"): profile[k] = request.form.get(k, "").strip()
    elif section == "security": flash("Password updated successfully.", "success"); session["user_profile"] = profile; return redirect(url_for("profile_page") + "#security")
    session["user_profile"] = profile; flash("Changes saved successfully!", "success"); return redirect(url_for("profile_page"))

@app.route("/api/clear-session", methods=["POST"])
def api_clear_session():
    profile, email = session.get("user_profile"), session.get("user_email")
    STATE.update({"dataset_name": None, "preprocessing": None, "graph": None, "graph_type": "undirected", "basic_stats": None, "advanced_stats": None, "centrality": None, "community_results": {}, "active_comm_alg": None, "community_graph": None, "community_details": None, "influence_results": {}, "active_influence": None, "scalability": None, "experiment_ts": None})
    if profile: session["user_profile"] = profile
    if email: session["user_email"] = email
    session["logged_in"] = True
    return jsonify({"status": "ok"})

@app.route("/")
def index(): return render_template("index.html", state=STATE)
@app.route("/dataset")
def dataset_page(): return render_template("dataset.html", state=STATE)
@app.route("/graph")
def graph_page(): return render_template("graph.html", state=STATE)
@app.route("/communities")
def communities_page(): return render_template("communities.html", state=STATE)
@app.route("/community-graph")
def community_graph_page(): return render_template("community_graph.html", state=STATE)
@app.route("/influence")
def influence_page(): return render_template("influence.html", state=STATE)
@app.route("/comparison")
def comparison_page(): return render_template("comparison.html", state=STATE)
@app.route("/campaign")
def campaign_page(): return render_template("campaign.html", state=STATE)
@app.route("/complexity")
def complexity_page(): return render_template("complexity.html", state=STATE)
@app.route("/reports")
def reports_page(): return render_template("reports.html", state=STATE, experiments=load_experiments(app.config["RESULTS_DIR"]), conclusion=generate_conclusion(STATE))

@app.route("/api/dataset/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files: return err("No file part in request.")
    f = request.files["file"]
    if not f.filename: return err("No file selected.")
    if not allowed_file(f.filename): return err("Unsupported file type. Use .csv, .txt, .edgelist, or .gz")
    filename, filepath = secure_filename(f.filename), os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename)); f.save(filepath)
    directed = request.form.get("directed", "false").lower() == "true"
    result = process_dataset(filepath); G = build_graph(result.get("edges", []), directed=directed)
    STATE.update(dataset_name=filename, preprocessing=result, graph=G, graph_type="directed" if directed else "undirected", basic_stats=compute_basic_stats(G), advanced_stats=None, centrality=None, community_results={}, active_comm_alg=None, community_graph=None, community_details=None, influence_results={}, active_influence=None, scalability=None, experiment_ts=int(time.time()))
    return ok(preprocessing=result)

@app.route("/api/dataset/generate", methods=["POST"])
def api_generate():
    data = request.get_json(silent=True) or {}; G = generate_network(int(data.get("num_nodes", 300)), data.get("topology", "scale_free")); result = {"unique_nodes": G.number_of_nodes(), "final_edges": G.number_of_edges(), "removed_duplicates": 0, "removed_self_loops": 0}
    STATE.update(dataset_name=f"synthetic_{data.get('topology','scale_free')}_{G.number_of_nodes()}", preprocessing=result, graph=G, graph_type="undirected", basic_stats=compute_basic_stats(G), advanced_stats=None, centrality=None, community_results={}, active_comm_alg=None, community_graph=None, community_details=None, influence_results={}, active_influence=None, scalability=None, experiment_ts=int(time.time()))
    return ok(preprocessing=result)

@app.route("/api/dataset/load-sample", methods=["POST"])
def api_load_sample():
    path = os.path.join(app.config["DATA_RAW_DIR"], "facebook_combined.txt")
    if not os.path.exists(path): return err("Sample dataset not found in data/raw/facebook_combined.txt")
    result = process_dataset(path); G = build_graph(result.get("edges", []), directed=False); STATE.update(dataset_name="facebook_combined.txt", preprocessing=result, graph=G, graph_type="undirected", basic_stats=compute_basic_stats(G), advanced_stats=None, centrality=None, community_results={}, active_comm_alg=None, community_graph=None, community_details=None, influence_results={}, active_influence=None, scalability=None, experiment_ts=int(time.time())); return ok(preprocessing=result)

@app.route("/api/graph/stats")
@graph_required
def api_graph_stats(): return ok(basic=STATE["basic_stats"], advanced=STATE["advanced_stats"], centrality=STATE["centrality"])

@app.route("/api/graph/advanced", methods=["POST"])
@graph_required
def api_graph_advanced(): STATE["advanced_stats"] = compute_advanced_stats(STATE["graph"]); return ok(advanced=STATE["advanced_stats"])

@app.route("/api/graph/centrality", methods=["POST"])
@graph_required
def api_centrality(): STATE["centrality"] = compute_centrality(STATE["graph"]); return ok(centrality=STATE["centrality"])

@app.route("/api/graph/viz")
@graph_required
def api_graph_viz(): return ok(viz=build_graph_viz(STATE["graph"]))

@app.route("/api/community/<algorithm>", methods=["POST"])
@graph_required
def api_community(algorithm):
    G = STATE["graph"]; funcs = {"greedy": run_greedy_modularity, "louvain": run_louvain, "girvan_newman": run_girvan_newman}
    if algorithm not in funcs: return err("Unknown community algorithm.")
    result = funcs[algorithm](G); STATE["community_results"][algorithm], STATE["active_comm_alg"] = result, algorithm; STATE["community_details"] = compute_community_details(G, result); return ok(result=result)

@app.route("/api/communities/overview")
@communities_required
def api_community_overview(): return ok(viz=build_community_overview_viz(STATE["graph"], STATE["community_results"].get(STATE["active_comm_alg"])))

@app.route("/api/community-graph")
@communities_required
def api_community_graph(): STATE["community_graph"] = build_community_graph(STATE["graph"], STATE["community_results"][STATE["active_comm_alg"]]); return ok(graph=build_graph_viz(STATE["community_graph"]))

@app.route("/api/influence/<method>", methods=["POST"])
@graph_required
def api_influence(method):
    data = request.get_json(silent=True) or {}; k = max(1, int(data.get("k", 5))); model = data.get("model", "ic"); p = float(data.get("probability", 0.1)); sims = max(1, int(data.get("simulations", 100)))
    simulator = simulate_ic if model == "ic" else simulate_lt
    if method == "greedy": result = run_greedy(STATE["graph"], k, simulator, sims, p)
    elif method == "celf": result = run_celf(STATE["graph"], k, simulator, sims, p)
    else: result = run_all_baselines(STATE["graph"], k, simulator, sims, p).get(method)
    if result is None: return err("Unknown influence method.")
    STATE["influence_results"][method], STATE["active_influence"] = result, method; return ok(result=result)

@app.route("/api/comparison", methods=["POST"])
@graph_required
def api_comparison():
    data = request.get_json(silent=True) or {}; return ok(comparison=run_all_baselines(STATE["graph"], int(data.get("k", 5))))

@app.route("/api/scalability", methods=["POST"])
def api_scalability(): STATE["scalability"] = run_scalability_analysis(); return ok(scalability=STATE["scalability"])

@app.route("/api/search-node")
@graph_required
def api_search_node():
    q = request.args.get("q", "").strip().lower(); matches = []
    for n in STATE["graph"].nodes:
        if q in str(n).lower(): matches.append(_get_node_info(STATE["graph"], n))
        if len(matches) >= 50: break
    return ok(results=matches, count=len(matches))

if __name__ == "__main__": app.run(host="0.0.0.0", port=5000, debug=True)
