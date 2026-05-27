#!/usr/bin/env python3
"""
panel_chatbot_v3.py
────────────────────
Panel HTML interactivo para los resultados de Analisis_Score_Chatbot_v3.py.

Diferencias respecto a versiones anteriores del panel:
  • Sin sección de InfraScore (eliminado en v3).
  • Sin toggle Base / +Domain (v3 produce un único resultado por modelo).
  • Sin columna Δ Score.
  • Fórmula del score actualizada: 5 componentes en lugar de 6.
  • Score components: 5 cards (DetectionScore, Coverage, DomainFit,
    RegulatoryFit, PenaltyFN).
  • Se mantienen todos los gráficos de riesgo y comparación por clase.

Uso:
    python panel_chatbot_v3.py chatbot_v3_results.json
    python panel_chatbot_v3.py chatbot_v3_results.json -o panel_v3.html
    python panel_chatbot_v3.py chatbot_v3_results.json --open
"""

import sys, subprocess

def _pip(*pkgs):
    subprocess.check_call([sys.executable,"-m","pip","install","--quiet",*pkgs],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Solo requiere la librería estándar de Python.
import json, os, argparse, webbrowser
from datetime import date


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def safe(v, d=0.0):
    return v if v is not None else d

def score_color_css(s):
    if s >= 75: return "var(--green)"
    if s >= 55: return "var(--accent2)"
    if s >= 35: return "var(--yellow)"
    if s >  0:  return "var(--orange)"
    return "var(--red)"

def risk_color_css(r):
    if r >= 0.9: return "var(--risk-critical)"
    if r >= 0.6: return "var(--risk-high)"
    if r >= 0.3: return "var(--risk-medium)"
    if r >  0:   return "var(--risk-low)"
    return "var(--risk-none)"

def risk_level(r):
    if r >= 0.9: return "Crítico"
    if r >= 0.6: return "Alto"
    if r >= 0.3: return "Medio"
    if r >  0:   return "Bajo"
    return "Sin riesgo"

def risk_css_cls(r):
    if r >= 0.9: return "risk-critical"
    if r >= 0.6: return "risk-high"
    if r >= 0.3: return "risk-medium"
    if r >  0:   return "risk-low"
    return "risk-none"

def interp_css_cls(interp):
    lo = interp.lower()
    if "recomendado" in lo: return "interp-optimal"
    if "operacionalmente" in lo: return "interp-baseline"
    if "mitigaciones" in lo: return "interp-limited"
    if "línea base" in lo or "complemento" in lo: return "interp-limited"
    return "interp-nofeasible"

PARADIGM_META = {
    "huggingface":       {"icon":"🤗","label":"HuggingFace NER","color":"rgba(245,158,11,0.85)","css":"p-hf"},
    "presidio_baseline": {"icon":"🛡️","label":"Presidio Baseline","color":"rgba(59,130,246,0.85)","css":"p-base"},
}

def pmeta(p):
    return PARADIGM_META.get(p, {"icon":"❓","label":p,"color":"rgba(136,146,176,0.7)","css":"p-unknown"})

def pb_cls(p):
    return {"huggingface":"pb-hf","presidio_baseline":"pb-base"}.get(p,"pb-unknown")


# ─── EXTRACCIÓN DE DATOS ──────────────────────────────────────────────────────

def extract_models(data: list) -> list:
    raw = []
    for r in data:
        if not isinstance(r, dict): continue
        mbl   = r.get("evaluation", {}).get("metrics_by_label", {})
        micro = r.get("evaluation", {}).get("micro_metrics",    {})
        crit  = r.get("critical_fn_contributions", {})
        classes = list(mbl.keys())

        raw.append({
            "name":          r.get("name","—"),
            "short":         r.get("name","—").replace("OpenMed-PII-","").replace("OpenMed-","").replace("Presidio-","Pres-"),
            "paradigm":      r.get("paradigm","huggingface"),
            "success":       bool(r.get("success",False)),
            "score_100":     round(safe(r.get("score_100")),4),
            "final_score":   round(safe(r.get("final_score")),8),
            "interpretation":r.get("interpretation","—"),
            "detection_score": round(safe(r.get("detection_score")),6),
            "coverage":      round(safe(r.get("coverage",1.0)),4),
            "domain_fit":    round(safe(r.get("domain_fit")),4),
            "regulatory_fit":round(safe(r.get("regulatory_fit")),4),
            "penalty_fn":    round(safe(r.get("penalty_fn")),6),
            "critical_fn":   round(safe(r.get("critical_fn")),6),
            # Sin InfraScore en v3
            "load_time":         round(safe(r.get("load_time")),3),
            "inference_latency": round(safe(r.get("inference_latency")),3),
            "ground_truth_count":   r.get("ground_truth_count",0),
            "prediction_count":     r.get("prediction_count",0),
            "out_of_domain_filtered": r.get("out_of_domain_filtered",0),
            "corpus_samples":    r.get("corpus_samples",0),
            "f2_micro":          round(safe(micro.get("f2")),4),
            "precision_micro":   round(safe(micro.get("precision")),4),
            "recall_micro":      round(safe(micro.get("recall")),4),
            "tp_total":          micro.get("tp",0),
            "fp_total":          micro.get("fp",0),
            "fn_total":          micro.get("fn",0),
            "all_eval_classes":  classes,
            "f2_by_class":      {c: round(safe(mbl[c].get("f2")),4) for c in classes},
            "weight_by_class":  {c: round(safe(mbl[c].get("criticality_weight")),6) for c in classes},
            "fn_rate_by_class": {c: round(safe(mbl[c].get("fn_rate")),4) for c in classes},
            "crit_fn_by_class": {c: {
                "severity_weight": round(safe(crit.get(c,{}).get("severity_weight")),6),
                "fn_rate":         round(safe(crit.get(c,{}).get("fn_rate")),4),
                "contribution":    round(safe(crit.get(c,{}).get("contribution")),6),
            } for c in classes},
            "r1": round(safe(r.get("r1_weakest_class_risk")),6),
            "r1_100":      round(safe(r.get("r1_weakest_class_risk_100")),2),
            "r1_class":    r.get("r1_weakest_class") or "—",
            "r2":          round(safe(r.get("r2_systemic_risk")),6),
            "r2_100":      round(safe(r.get("r2_systemic_risk_100")),2),
            "r_final":     round(safe(r.get("r_final")),6),
            "r_final_100": round(safe(r.get("r_final_100")),2),
            "coverage_basis": r.get("coverage_basis","—"),
        })

    raw.sort(key=lambda x: x["score_100"], reverse=True)
    for i, m in enumerate(raw): m["rank"] = i+1
    return raw


def extract_all_classes(models):
    seen, ordered = set(), []
    for m in models:
        for c in m["all_eval_classes"]:
            if c not in seen:
                ordered.append(c); seen.add(c)
    return ordered


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
:root{
  --bg:#0f1117;--surface:#1a1d27;--surface2:#222638;--border:#2e3350;
  --text:#e2e8f0;--muted:#8892b0;--accent:#7c6af7;--accent2:#5eead4;
  --green:#22d3a5;--yellow:#f59e0b;--red:#ef4444;--orange:#f97316;
  --blue:#3b82f6;--purple:#a855f7;--teal:#14b8a6;
  --risk-critical:#ef4444;--risk-high:#f97316;
  --risk-medium:#f59e0b;--risk-low:#22d3a5;--risk-none:#8892b0;
  --p-hf:#f59e0b;--p-base:#3b82f6;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;}

.header{background:linear-gradient(135deg,#1a1d27,#16192a);border-bottom:1px solid var(--border);padding:28px 40px 24px;}
.header-top{display:flex;align-items:center;gap:16px;margin-bottom:6px;}
.shield-icon{width:44px;height:44px;background:linear-gradient(135deg,var(--teal),#ef4444);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;}
.header h1{font-size:22px;font-weight:700;}
.header p{font-size:13px;color:var(--muted);margin-top:2px;}
.meta-pills{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;}
.pill{font-size:11.5px;padding:4px 12px;border-radius:20px;border:1px solid var(--border);color:var(--muted);background:var(--surface);}
.pill span{color:var(--accent2);font-weight:600;}
.pill.rpill span{color:var(--risk-high);}

.main{padding:32px 40px;max-width:1380px;margin:0 auto;}
.section-title{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:16px;margin-top:36px;}
.section-title:first-of-type{margin-top:0;}

/* BEST CARD */
.best-card{background:linear-gradient(135deg,#121e2a,#0f1e1a);border:1px solid #1e4a40;border-radius:16px;padding:24px 28px;display:flex;align-items:center;gap:28px;margin-bottom:8px;position:relative;overflow:hidden;}
.best-card::before{content:'';position:absolute;top:-40px;right:-40px;width:200px;height:200px;background:radial-gradient(circle,rgba(20,184,166,.2),transparent 70%);}
.best-badge{background:linear-gradient(135deg,var(--teal),var(--accent));border-radius:12px;padding:12px 18px;text-align:center;flex-shrink:0;}
.best-badge .score-big{font-size:38px;font-weight:800;color:#fff;line-height:1;}
.best-badge .score-label{font-size:11px;color:rgba(255,255,255,.75);margin-top:2px;}
.best-info h2{font-size:18px;font-weight:700;}
.best-kpis{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;}
.kpi-val{font-size:14px;font-weight:700;color:var(--accent2);}
.kpi-lbl{font-size:10.5px;color:var(--muted);}

/* PARADIGM BADGES */
.paradigm-badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;white-space:nowrap;}
.pb-hf{background:rgba(245,158,11,.12);color:var(--p-hf);border:1px solid rgba(245,158,11,.35);}
.pb-base{background:rgba(59,130,246,.12);color:var(--p-base);border:1px solid rgba(59,130,246,.35);}
.pb-unknown{background:rgba(136,146,176,.1);color:var(--muted);border:1px solid rgba(136,146,176,.25);}

/* RISK STRIP */
.risk-strip{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:32px;margin-top:12px;}
.rstrip-card{flex:1;min-width:155px;border-radius:14px;padding:15px 16px;border:1px solid;cursor:pointer;}
.rc-card{background:rgba(239,68,68,.09);border-color:rgba(239,68,68,.3);}
.rh-card{background:rgba(249,115,22,.09);border-color:rgba(249,115,22,.3);}
.rm-card{background:rgba(245,158,11,.09);border-color:rgba(245,158,11,.3);}
.rl-card{background:rgba(34,211,165,.06);border-color:rgba(34,211,165,.25);}
.rn-card{background:rgba(136,146,176,.05);border-color:rgba(136,146,176,.2);}
.rs-rank{font-size:11px;color:var(--muted);margin-bottom:3px;}
.rs-pct{font-size:26px;font-weight:900;line-height:1;}
.rs-sub{font-size:10px;font-weight:600;margin-top:2px;}
.rs-bar-wrap{height:4px;background:rgba(255,255,255,.06);border-radius:2px;margin-top:7px;}
.rs-bar{height:100%;border-radius:2px;}
.rcc{color:var(--risk-critical);}.rbc{background:var(--risk-critical);}
.rch{color:var(--risk-high);}    .rbh{background:var(--risk-high);}
.rcm{color:var(--risk-medium);}  .rbm{background:var(--risk-medium);}
.rcl{color:var(--risk-low);}     .rbl{background:var(--risk-low);}
.rcn{color:var(--risk-none);}    .rbn{background:var(--risk-none);}

/* TABLE */
.table-wrap{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden;margin-bottom:8px;}
table{width:100%;border-collapse:collapse;}
thead th{background:var(--surface2);padding:10px 12px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);text-align:left;border-bottom:1px solid var(--border);}
tbody tr{border-bottom:1px solid var(--border);transition:background .15s;cursor:pointer;}
tbody tr:last-child{border-bottom:none;}
tbody tr:hover{background:rgba(20,184,166,.05);}
tbody tr.selected{background:rgba(20,184,166,.1);}
tbody td{padding:10px 12px;font-size:12px;vertical-align:middle;}
.rank-badge{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:8px;font-weight:800;font-size:13px;}
.rk1{background:linear-gradient(135deg,#ffd700,#f59e0b);color:#000;}
.rk2{background:linear-gradient(135deg,#c0c0c0,#94a3b8);color:#000;}
.rk3{background:linear-gradient(135deg,#cd7f32,#b45309);color:#fff;}
.rko{background:var(--surface2);color:var(--muted);}
.mn{font-weight:600;}
.mn small{display:block;font-size:10px;color:var(--muted);font-weight:400;margin-top:1px;}
.sbar-wrap{height:4px;background:var(--surface2);border-radius:2px;margin-top:4px;width:90px;}
.sbar{height:100%;border-radius:2px;}
.rbar-wrap{height:4px;background:var(--surface2);border-radius:2px;margin-top:4px;width:80px;}
.rbar{height:100%;border-radius:2px;}

/* BADGES */
.interp-badge{display:inline-block;font-size:11px;font-weight:600;padding:4px 10px;border-radius:20px;white-space:nowrap;}
.interp-optimal{background:rgba(34,211,165,.12);color:var(--green);border:1px solid rgba(34,211,165,.3);}
.interp-baseline{background:rgba(94,234,212,.1);color:var(--accent2);border:1px solid rgba(94,234,212,.3);}
.interp-limited{background:rgba(245,158,11,.15);color:var(--yellow);border:1px solid rgba(245,158,11,.3);}
.interp-nofeasible{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.25);}
.risk-badge{display:inline-block;font-size:11px;font-weight:700;padding:4px 10px;border-radius:20px;white-space:nowrap;}
.risk-critical{background:rgba(239,68,68,.15);color:var(--risk-critical);border:1px solid rgba(239,68,68,.35);}
.risk-high{background:rgba(249,115,22,.15);color:var(--risk-high);border:1px solid rgba(249,115,22,.35);}
.risk-medium{background:rgba(245,158,11,.15);color:var(--risk-medium);border:1px solid rgba(245,158,11,.35);}
.risk-low{background:rgba(34,211,165,.12);color:var(--risk-low);border:1px solid rgba(34,211,165,.3);}
.risk-none{background:rgba(136,146,176,.1);color:var(--muted);border:1px solid rgba(136,146,176,.25);}

/* DETAIL PANEL */
.detail-panel{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden;margin-bottom:8px;}
.dp-header{background:var(--surface2);padding:15px 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.dp-header h3{font-size:15px;font-weight:700;flex:1;}
.dp-body{padding:22px;}
.click-hint{font-size:12px;color:var(--muted);margin-bottom:10px;padding:7px 13px;background:var(--surface2);border-radius:8px;display:inline-flex;align-items:center;gap:6px;}

/* HERO ROW */
.hero-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px;}
.hero-box{border-radius:14px;padding:16px 20px;}
.score-hbox{background:linear-gradient(135deg,#0f1e1a,#0f1221);border:1px solid var(--teal);}
.risk-hbox{border:1px solid;}
.hlbl{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:7px;}
.hbig{font-size:46px;font-weight:900;line-height:1;}
.score-grad{background:linear-gradient(135deg,var(--teal),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hunit{font-size:12px;color:var(--muted);margin-top:2px;}
.ibig{font-size:13px;font-weight:700;padding:7px 14px;border-radius:8px;display:inline-block;margin-top:8px;}
.risk-sub-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;}
.rsub-card{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:10px;padding:12px 14px;}
.rsub-val{font-size:20px;font-weight:800;}
.rsub-lbl{font-size:10.5px;color:var(--muted);margin-top:3px;}

/* FORMULA */
.formula-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;}
.fc{background:rgba(20,184,166,.06);border:1px solid rgba(20,184,166,.2);border-radius:10px;padding:11px 14px;font-size:11.5px;color:var(--muted);}
.fc.rfc{background:rgba(239,68,68,.06);border-color:rgba(239,68,68,.2);}
.fc strong{color:var(--teal);}
.fc.rfc strong{color:var(--risk-high);}

/* SCORE COMPONENTS — 5 cards (sin InfraScore) */
.comps5{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px;}
.comp-card{background:var(--surface2);border:1px solid var(--border);border-radius:11px;padding:13px;}
.cval{font-size:20px;font-weight:800;color:var(--accent2);}
.clbl{font-size:10.5px;color:var(--muted);margin-top:4px;}
.cbar-wrap{height:4px;background:var(--border);border-radius:2px;margin-top:8px;}
.cbar{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--teal),var(--accent));}

/* CRITICAL FN TABLE */
.dtitle{font-size:13px;font-weight:700;color:var(--text);margin-bottom:10px;}
.dtable{width:100%;border-collapse:collapse;margin-bottom:20px;}
.dtable th{background:var(--surface2);padding:8px 11px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-align:left;border-bottom:1px solid var(--border);}
.dtable td{padding:8px 11px;font-size:12px;border-bottom:1px solid rgba(46,51,80,.4);}
.dtable tr:last-child td{border-bottom:none;}
.mini-bar-wrap{height:5px;background:rgba(255,255,255,.06);border-radius:3px;}
.mini-bar{height:100%;border-radius:3px;}

/* CHART */
.ctitle{font-size:13px;font-weight:700;color:var(--text);margin-bottom:11px;}
.chart-wrap{position:relative;height:260px;}
.chart-wrap.tall{height:320px;}
.chart-wrap.xtall{height:400px;}

/* MICRO */
.micro-row{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;}
.mc{flex:1;min-width:90px;background:var(--surface2);border:1px solid var(--border);border-radius:11px;padding:12px;text-align:center;}
.mcv{font-size:20px;font-weight:800;}
.mcl{font-size:10px;color:var(--muted);margin-top:3px;}
.mc-f2{color:var(--accent2);}  .mc-pr{color:var(--purple);}  .mc-rc{color:var(--blue);}
.mc-tp{color:var(--green);}    .mc-fp{color:var(--orange);}  .mc-fn{color:var(--red);}
.mc-filt{color:var(--muted);}

/* LAT ROW */
.lat-row{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap;}
.lcard{flex:1;min-width:100px;background:var(--surface2);border:1px solid var(--border);border-radius:11px;padding:12px;}
.lval{font-size:19px;font-weight:800;color:var(--blue);}
.llbl{font-size:10.5px;color:var(--muted);margin-top:3px;}

.divider{border:none;border-top:1px solid var(--border);margin:6px 0 20px;}
.footer{text-align:center;padding:22px 40px;font-size:12px;color:var(--muted);border-top:1px solid var(--border);}

@media(max-width:960px){
  .main,.header{padding:20px 14px;}
  .comps5{grid-template-columns:repeat(3,1fr);}
  .hero-row,.formula-row{grid-template-columns:1fr;}
  .best-card{flex-direction:column;align-items:flex-start;}
}
"""


# ─── HTML BUILDER ─────────────────────────────────────────────────────────────

def _rstrip_cls(r):
    if r >= 0.9: return "rc-card","c"
    if r >= 0.6: return "rh-card","h"
    if r >= 0.3: return "rm-card","m"
    if r >  0:   return "rl-card","l"
    return "rn-card","n"

def _risk_hero_style(r):
    styles = {
        "c":("rgba(239,68,68,.1)",  "rgba(239,68,68,.4)"),
        "h":("rgba(249,115,22,.1)", "rgba(249,115,22,.4)"),
        "m":("rgba(245,158,11,.1)", "rgba(245,158,11,.4)"),
        "l":("rgba(34,211,165,.07)","rgba(34,211,165,.35)"),
        "n":("rgba(136,146,176,.06)","rgba(136,146,176,.25)"),
    }
    k  = "c" if r>=0.9 else "h" if r>=0.6 else "m" if r>=0.3 else "l" if r>0 else "n"
    bg, bd = styles[k]
    return f"background:{bg};border-color:{bd};"


def build_html(models: list, source_file: str) -> str:
    today    = date.today().strftime("%-d de %B de %Y")
    n        = len(models)
    best     = models[0] if models else {}
    all_cls  = extract_all_classes(models)
    cls_lbl  = [c.replace("_"," ").title() for c in all_cls]

    # ── Risk strip ────────────────────────────────────────────────────────────
    strip_html = ""
    for m in models:
        ccard, csuf = _rstrip_cls(m["r_final"])
        bw = int(m["r_final_100"])
        pm = pmeta(m["paradigm"])
        strip_html += f"""
        <div class="rstrip-card {ccard}" onclick="showDetail({m['rank']-1})">
          <div class="rs-rank">#{m['rank']} · <span style="color:{pm['color']}">{pm['icon']} {m['short']}</span></div>
          <div class="rs-pct rc{csuf}">{m['r_final_100']:.1f}%</div>
          <div class="rs-sub rc{csuf}">{risk_level(m['r_final'])} riesgo</div>
          <div class="rs-sub" style="color:var(--muted);font-size:10px;margin-top:2px">
            Score: {m['score_100']:.1f} · R1: {m['r1_100']:.1f}%
          </div>
          <div class="rs-bar-wrap"><div class="rs-bar rb{csuf}" style="width:{bw}%"></div></div>
        </div>"""

    # ── Ranking rows ──────────────────────────────────────────────────────────
    rows_html = ""
    for m in models:
        rk_cls = {1:"rk1",2:"rk2",3:"rk3"}.get(m["rank"],"rko")
        sc = score_color_css(m["score_100"])
        rc = risk_color_css(m["r_final"])
        ic = interp_css_cls(m["interpretation"])
        ric= risk_css_cls(m["r_final"])
        pb = pb_cls(m["paradigm"])
        pm = pmeta(m["paradigm"])
        sw = min(100, int(m["score_100"]))
        rw = int(m["r_final_100"])
        r1w= int(m["r1_100"])
        rows_html += f"""
        <tr data-idx="{m['rank']-1}">
          <td><span class="rank-badge {rk_cls}">{m['rank']}</span></td>
          <td><div class="mn">{m['short']}<small>{m['name']}</small></div></td>
          <td><span class="paradigm-badge {pb}">{pm['icon']} {pm['label']}</span></td>
          <td>
            <div style="font-size:17px;font-weight:800;color:{sc}">{m['score_100']:.2f}</div>
            <div class="sbar-wrap"><div class="sbar" style="width:{sw}%;background:{sc}"></div></div>
          </td>
          <td><span class="interp-badge {ic}">{m['interpretation']}</span></td>
          <td>
            <div style="font-size:16px;font-weight:800;color:{rc}">{m['r_final_100']:.1f}%</div>
            <div class="rbar-wrap"><div class="rbar" style="width:{rw}%;background:{rc}"></div></div>
          </td>
          <td><span class="risk-badge {ric}">{risk_level(m['r_final'])}</span></td>
          <td>
            <div style="font-size:12px;color:var(--risk-high)">{m['r1_100']:.1f}%</div>
            <div class="rbar-wrap"><div class="rbar" style="width:{r1w}%;background:var(--risk-high)"></div></div>
          </td>
          <td style="font-size:11px;color:var(--muted)">{m['r1_class'].replace('_',' ')}</td>
          <td style="color:var(--accent2);font-weight:700">{m['detection_score']:.4f}</td>
          <td style="font-weight:600">{m['f2_micro']:.3f}</td>
        </tr>"""

    # ── JS data ───────────────────────────────────────────────────────────────
    js_models = json.dumps([{
        "rank":m["rank"],"name":m["name"],"short":m["short"],
        "paradigm":m["paradigm"],
        "final_score":m["final_score"],"score_100":m["score_100"],
        "interpretation":m["interpretation"],
        "detection_score":m["detection_score"],"coverage":m["coverage"],
        "domain_fit":m["domain_fit"],"regulatory_fit":m["regulatory_fit"],
        "penalty_fn":m["penalty_fn"],"critical_fn":m["critical_fn"],
        "load_time":m["load_time"],"inference_latency":m["inference_latency"],
        "ground_truth_count":m["ground_truth_count"],
        "prediction_count":m["prediction_count"],
        "out_of_domain_filtered":m["out_of_domain_filtered"],
        "corpus_samples":m["corpus_samples"],
        "f2_micro":m["f2_micro"],"precision_micro":m["precision_micro"],
        "recall_micro":m["recall_micro"],
        "tp_total":m["tp_total"],"fp_total":m["fp_total"],"fn_total":m["fn_total"],
        "f2_by_class":{c:m["f2_by_class"].get(c,0.0) for c in all_cls},
        "weight_by_class":{c:m["weight_by_class"].get(c,0.0) for c in all_cls},
        "fn_rate_by_class":{c:m["fn_rate_by_class"].get(c,0.0) for c in all_cls},
        "crit_fn_by_class":{c:m["crit_fn_by_class"].get(c,{"severity_weight":0,"fn_rate":0,"contribution":0}) for c in all_cls},
        "r1":m["r1"],"r1_100":m["r1_100"],"r1_class":m["r1_class"],
        "r2":m["r2"],"r2_100":m["r2_100"],
        "r_final":m["r_final"],"r_final_100":m["r_final_100"],
        "coverage_basis":m["coverage_basis"],
    } for m in models], ensure_ascii=False, indent=2)

    P_COLORS = json.dumps({
        "huggingface":       "rgba(245,158,11,0.80)",
        "presidio_baseline": "rgba(59,130,246,0.80)",
    })
    js_classes = json.dumps(all_cls, ensure_ascii=False)
    js_clslbl  = json.dumps(cls_lbl, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Panel v3 — Privacy Detection Score (sin InfraScore)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>{CSS}</style>
</head>
<body>

<div class="header">
  <div class="header-top">
    <div class="shield-icon">🛡️</div>
    <div>
      <h1>Panel v3 — Domain-Adjusted Privacy Detection Score</h1>
      <p>Score = DetectionScore × Coverage × DomainFit × RegulatoryFit × PenaltyFN
         &nbsp;·&nbsp; Sin InfraScore · Sin capas REGEX/NER · {os.path.basename(source_file)}</p>
    </div>
  </div>
  <div class="meta-pills">
    <div class="pill">Modelos: <span>{n}</span></div>
    <div class="pill">Clases evaluadas: <span>{len(all_cls)}</span></div>
    <div class="pill">Mejor score: <span>{best.get('score_100',0):.2f}/100</span></div>
    <div class="pill rpill">R_final mín: <span>{min(m['r_final_100'] for m in models):.1f}%</span></div>
  </div>
</div>

<div class="main">

<!-- MEJOR MODELO -->
<div class="section-title">🏆 Mejor modelo</div>
<div class="best-card">
  <div class="best-badge">
    <div class="score-big">{best.get('score_100',0):.2f}</div>
    <div class="score-label">Score / 100</div>
  </div>
  <div class="best-info">
    <h2>{best.get('name','—')}</h2>
    <span class="paradigm-badge {pb_cls(best.get('paradigm',''))}">{pmeta(best.get('paradigm','')).get('icon','')} {pmeta(best.get('paradigm','')).get('label','')}</span>
    <span class="interp-badge {interp_css_cls(best.get('interpretation',''))}" style="margin-left:8px">{best.get('interpretation','—')}</span>
    <div class="best-kpis">
      <div><span class="kpi-val">{best.get('detection_score',0):.4f}</span><div class="kpi-lbl">DetectionScore</div></div>
      <div><span class="kpi-val">{best.get('coverage',0):.3f}</span><div class="kpi-lbl">Coverage</div></div>
      <div><span class="kpi-val">{best.get('penalty_fn',0):.4f}</span><div class="kpi-lbl">PenaltyFN</div></div>
      <div><span class="kpi-val" style="color:{risk_color_css(best.get('r_final',0))}">{best.get('r_final_100',0):.1f}%</span><div class="kpi-lbl">R_final</div></div>
      <div><span class="kpi-val" style="color:var(--accent2)">{best.get('f2_micro',0):.3f}</span><div class="kpi-lbl">F2 micro</div></div>
    </div>
  </div>
</div>

<!-- RISK STRIP -->
<div class="section-title">⚠️ Resumen de riesgo</div>
<div class="risk-strip">{strip_html}</div>

<!-- RANKING TABLE -->
<div class="section-title">📊 Ranking completo</div>
<p class="click-hint">👆 Clic en fila para análisis detallado</p>
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>#</th><th>Modelo</th><th>Paradigma</th>
        <th>Score (/100)</th><th>Interpretación</th>
        <th>R_final</th><th>Nivel riesgo</th>
        <th>R1 Weakest</th><th>Clase débil</th>
        <th>DetectionScore</th><th>F2 micro</th>
      </tr>
    </thead>
    <tbody id="ranking-tbody">{rows_html}</tbody>
  </table>
</div>

<!-- DETAIL PANEL -->
<div class="section-title">🔍 Análisis detallado</div>
<div class="detail-panel" id="detail-panel">
  <div class="dp-header">
    <h3 id="d-name">—</h3>
    <span id="d-pbadge" class="paradigm-badge">—</span>
    <span id="d-ibadge" class="interp-badge" style="margin-left:6px">—</span>
    <span id="d-rbadge" class="risk-badge"   style="margin-left:6px">—</span>
  </div>
  <div class="dp-body">

    <div class="hero-row">
      <div class="hero-box score-hbox">
        <div class="hlbl">Score de desempeño</div>
        <div class="hbig score-grad" id="d-score100">—</div>
        <div class="hunit">puntos de 100</div>
        <span class="ibig" id="d-ibig">—</span>
      </div>
      <div class="hero-box risk-hbox" id="d-risk-hbox">
        <div class="hlbl">Riesgo final (R_final)</div>
        <div class="hbig" id="d-rfinal-pct">—</div>
        <div class="hunit">riesgo combinado</div>
        <div class="risk-sub-row">
          <div class="rsub-card">
            <div class="rsub-val" id="d-r1-pct" style="color:var(--risk-high)">—</div>
            <div class="rsub-lbl">R1 — Clase más débil</div>
            <div style="font-size:10px;color:var(--muted);margin-top:4px">Clase: <b id="d-r1-class" style="color:var(--accent2)">—</b></div>
          </div>
          <div class="rsub-card">
            <div class="rsub-val" id="d-r2-pct" style="color:var(--risk-medium)">—</div>
            <div class="rsub-lbl">R2 — Riesgo sistémico</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Fórmula v3 (5 componentes) -->
    <div class="formula-row">
      <div class="fc">
        <strong>Score v3</strong> = DetectionScore × Coverage × DomainFit × RegulatoryFit × PenaltyFN<br>
        <strong id="f-det">—</strong> × <strong id="f-cov">—</strong> × <strong id="f-dom">—</strong> ×
        <strong id="f-reg">—</strong> × <strong id="f-pen">—</strong>
        → <strong id="f-res">—</strong>
      </div>
      <div class="fc rfc">
        <strong>R_final</strong> = 1 − (1 − R1) × (1 − R2)<br>
        R1: <strong id="f-r1">—</strong> &nbsp;|&nbsp; R2: <strong id="f-r2">—</strong>
        → <strong id="f-rf">—</strong>
      </div>
    </div>

    <!-- 5 componentes (sin InfraScore) -->
    <div class="ctitle">Componentes del score (v3 — sin InfraScore)</div>
    <div class="comps5">
      <div class="comp-card"><div class="cval" id="c-det">—</div><div class="clbl">DetectionScore</div><div class="cbar-wrap"><div class="cbar" id="cb-det"></div></div></div>
      <div class="comp-card"><div class="cval" id="c-cov">—</div><div class="clbl">Coverage</div><div class="cbar-wrap"><div class="cbar" id="cb-cov"></div></div></div>
      <div class="comp-card"><div class="cval" id="c-dom">—</div><div class="clbl">DomainFit</div><div class="cbar-wrap"><div class="cbar" id="cb-dom"></div></div></div>
      <div class="comp-card"><div class="cval" id="c-reg">—</div><div class="clbl">RegulatoryFit</div><div class="cbar-wrap"><div class="cbar" id="cb-reg"></div></div></div>
      <div class="comp-card"><div class="cval" id="c-pen">—</div><div class="clbl">PenaltyFN</div><div class="cbar-wrap"><div class="cbar" id="cb-pen"></div></div></div>
    </div>

    <!-- Métricas micro -->
    <div class="ctitle">Métricas micro del corpus</div>
    <div class="micro-row">
      <div class="mc"><div class="mcv mc-f2" id="m-f2">—</div><div class="mcl">F2 micro</div></div>
      <div class="mc"><div class="mcv mc-pr" id="m-pr">—</div><div class="mcl">Precision</div></div>
      <div class="mc"><div class="mcv mc-rc" id="m-rc">—</div><div class="mcl">Recall</div></div>
      <div class="mc"><div class="mcv mc-tp" id="m-tp">—</div><div class="mcl">TP</div></div>
      <div class="mc"><div class="mcv mc-fp" id="m-fp">—</div><div class="mcl">FP</div></div>
      <div class="mc"><div class="mcv mc-fn" id="m-fn">—</div><div class="mcl">FN</div></div>
      <div class="mc"><div class="mcv mc-filt" id="m-filt">—</div><div class="mcl">Filtradas (fuera dom.)</div></div>
    </div>

    <hr class="divider">

    <!-- Critical FN table -->
    <div class="dtitle">🎯 Contribuciones de riesgo crítico por clase</div>
    <table class="dtable" id="crit-table">
      <thead><tr><th>Clase PII</th><th>Severity ρ</th><th>FN Rate</th><th>Contribución</th><th>Visual</th></tr></thead>
      <tbody id="crit-tbody"></tbody>
    </table>

    <hr class="divider">

    <!-- F2 + FN Rate por clase -->
    <div class="ctitle">F2 y FN Rate por clase PII</div>
    <div class="chart-wrap tall"><canvas id="classChart"></canvas></div>

    <div class="lat-row">
      <div class="lcard"><div class="lval" id="l-load">—</div><div class="llbl">⏱ Carga modelo</div></div>
      <div class="lcard"><div class="lval" id="l-inf">—</div><div class="llbl">⚡ Latencia total</div></div>
      <div class="lcard"><div class="lval" id="l-gt">—</div><div class="llbl">✅ GT entidades</div></div>
      <div class="lcard"><div class="lval" id="l-pred">—</div><div class="llbl">🔎 Predicciones dom.</div></div>
      <div class="lcard"><div class="lval" id="l-samples">—</div><div class="llbl">📄 Turnos corpus</div></div>
    </div>
  </div>
</div>

<!-- SCATTER: Score vs R_final -->
<div class="section-title">📈 Score vs. Riesgo</div>
<div class="detail-panel">
  <div class="dp-body"><div class="chart-wrap xtall"><canvas id="scatterChart"></canvas></div></div>
</div>

<!-- F2 por clase -->
<div class="section-title">📊 F2 por clase — comparación</div>
<div class="detail-panel">
  <div class="dp-body"><div class="chart-wrap xtall"><canvas id="compF2Chart"></canvas></div></div>
</div>

<!-- FN Rate por clase -->
<div class="section-title">🔥 FN Rate por clase — comparación</div>
<div class="detail-panel">
  <div class="dp-body"><div class="chart-wrap xtall"><canvas id="compFNChart"></canvas></div></div>
</div>

</div>
<div class="footer">
  Panel generado el {today} &nbsp;·&nbsp;
  Domain-Adjusted Privacy Detection Score v3 &nbsp;·&nbsp;
  Sin InfraScore · Sin capas REGEX/NER
</div>

<script>
const MODELS   = {js_models};
const CLASSES  = {js_classes};
const CLS_LBL  = {js_clslbl};
const P_COLORS = {P_COLORS};

function fmt(v,d=4){{ return (v!=null&&!isNaN(v))?(+v).toFixed(d):'—'; }}
function scoreColor(s){{ return s>=75?'var(--green)':s>=55?'var(--accent2)':s>=35?'var(--yellow)':s>0?'var(--orange)':'var(--red)'; }}
function riskColor(r){{  return r>=0.9?'var(--risk-critical)':r>=0.6?'var(--risk-high)':r>=0.3?'var(--risk-medium)':r>0?'var(--risk-low)':'var(--risk-none)'; }}
function riskLevel(r){{  return r>=0.9?'Crítico':r>=0.6?'Alto':r>=0.3?'Medio':r>0?'Bajo':'Sin riesgo'; }}
function riskCls(r){{    return r>=0.9?'risk-critical':r>=0.6?'risk-high':r>=0.3?'risk-medium':r>0?'risk-low':'risk-none'; }}
function interpCls(i){{
  const lo=i.toLowerCase();
  if(lo.includes('recomendado')) return 'interp-optimal';
  if(lo.includes('operacionalmente')) return 'interp-baseline';
  if(lo.includes('mitigaciones')||lo.includes('complemento')) return 'interp-limited';
  return 'interp-nofeasible';
}}
function riskHeroStyle(r){{
  if(r>=0.9) return 'background:rgba(239,68,68,.1);border-color:rgba(239,68,68,.4);';
  if(r>=0.6) return 'background:rgba(249,115,22,.1);border-color:rgba(249,115,22,.4);';
  if(r>=0.3) return 'background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.4);';
  if(r>0)    return 'background:rgba(34,211,165,.07);border-color:rgba(34,211,165,.35);';
  return 'background:rgba(136,146,176,.06);border-color:rgba(136,146,176,.25);';
}}
function pbCls(p){{ return {{huggingface:'pb-hf',presidio_baseline:'pb-base'}}[p]||'pb-unknown'; }}
function pbLabel(p){{ return {{huggingface:'🤗 HuggingFace NER',presidio_baseline:'🛡️ Presidio Baseline'}}[p]||p; }}

document.querySelectorAll('#ranking-tbody tr').forEach(tr=>{{
  tr.addEventListener('click',()=>showDetail(+tr.dataset.idx));
}});

let classChartInst=null;

function showDetail(idx){{
  document.querySelectorAll('#ranking-tbody tr').forEach(r=>r.classList.remove('selected'));
  const row=document.querySelector(`tr[data-idx="${{idx}}"]`);
  if(row) row.classList.add('selected');
  const m=MODELS[idx];

  document.getElementById('d-name').textContent=m.name;
  const pb=document.getElementById('d-pbadge');
  pb.textContent=pbLabel(m.paradigm); pb.className='paradigm-badge '+pbCls(m.paradigm);
  const ib=document.getElementById('d-ibadge');
  ib.textContent=m.interpretation; ib.className='interp-badge '+interpCls(m.interpretation);
  const rb=document.getElementById('d-rbadge');
  rb.textContent=riskLevel(m.r_final)+' · '+fmt(m.r_final_100,1)+'%';
  rb.className='risk-badge '+riskCls(m.r_final);

  document.getElementById('d-score100').textContent=fmt(m.score_100,2);
  const dib=document.getElementById('d-ibig');
  dib.textContent=m.interpretation; dib.className='ibig '+interpCls(m.interpretation);

  const rc=riskColor(m.r_final);
  document.getElementById('d-rfinal-pct').textContent=fmt(m.r_final_100,1)+'%';
  document.getElementById('d-rfinal-pct').style.color=rc;
  document.getElementById('d-risk-hbox').style.cssText=riskHeroStyle(m.r_final);
  document.getElementById('d-r1-pct').textContent=fmt(m.r1_100,1)+'%';
  document.getElementById('d-r1-class').textContent=(m.r1_class||'—').replace(/_/g,' ');
  document.getElementById('d-r2-pct').textContent=fmt(m.r2_100,1)+'%';

  document.getElementById('f-det').textContent=fmt(m.detection_score,4);
  document.getElementById('f-cov').textContent=fmt(m.coverage,4);
  document.getElementById('f-dom').textContent=fmt(m.domain_fit,2);
  document.getElementById('f-reg').textContent=fmt(m.regulatory_fit,2);
  document.getElementById('f-pen').textContent=fmt(m.penalty_fn,4);
  document.getElementById('f-res').textContent=fmt(m.final_score,6);
  document.getElementById('f-r1').textContent=fmt(m.r1_100,1)+'%';
  document.getElementById('f-r2').textContent=fmt(m.r2_100,1)+'%';
  document.getElementById('f-rf').textContent=fmt(m.r_final_100,1)+'%';

  // 5 componentes (sin InfraScore)
  [['c-det','cb-det',m.detection_score],['c-cov','cb-cov',m.coverage],
   ['c-dom','cb-dom',m.domain_fit],['c-reg','cb-reg',m.regulatory_fit],
   ['c-pen','cb-pen',m.penalty_fn]].forEach(([v,b,val])=>{{
    const n=val||0;
    document.getElementById(v).textContent=fmt(n,4);
    document.getElementById(b).style.width=(n*100)+'%';
  }});

  document.getElementById('m-f2').textContent=fmt(m.f2_micro,3);
  document.getElementById('m-pr').textContent=fmt(m.precision_micro,3);
  document.getElementById('m-rc').textContent=fmt(m.recall_micro,3);
  document.getElementById('m-tp').textContent=m.tp_total||0;
  document.getElementById('m-fp').textContent=m.fp_total||0;
  document.getElementById('m-fn').textContent=m.fn_total||0;
  document.getElementById('m-filt').textContent=m.out_of_domain_filtered||0;
  document.getElementById('l-load').textContent=fmt(m.load_time,2)+'s';
  document.getElementById('l-inf').textContent=fmt(m.inference_latency,2)+'s';
  document.getElementById('l-gt').textContent=m.ground_truth_count||0;
  document.getElementById('l-pred').textContent=m.prediction_count||0;
  document.getElementById('l-samples').textContent=m.corpus_samples||0;

  const ctbody=document.getElementById('crit-tbody');
  ctbody.innerHTML='';
  CLASSES.forEach(c=>{{
    const cf=m.crit_fn_by_class[c]||{{}};
    const sw=cf.severity_weight||0; const fnr=cf.fn_rate||0; const cont=cf.contribution||0;
    if(sw===0&&fnr===0) return;
    const cc=cont>0.1?'var(--risk-critical)':cont>0?'var(--risk-high)':'var(--green)';
    const fc=fnr>=1?'var(--risk-critical)':fnr>0?'var(--risk-high)':'var(--green)';
    ctbody.innerHTML+=`
      <tr>
        <td style="font-weight:600">${{c.replace(/_/g,' ')}}</td>
        <td style="color:var(--accent2)">${{(sw*100).toFixed(1)}}%</td>
        <td style="color:${{fc}};font-weight:700">${{(fnr*100).toFixed(0)}}%</td>
        <td style="color:${{cc}};font-weight:700">${{cont.toFixed(4)}}</td>
        <td><div class="mini-bar-wrap"><div class="mini-bar" style="width:${{Math.min(100,(cont*500).toFixed(0))}}%;background:${{cc}}"></div></div></td>
      </tr>`;
  }});
  if(!ctbody.innerHTML)
    ctbody.innerHTML='<tr><td colspan="5" style="color:var(--green);text-align:center;padding:12px">✅ Sin contribuciones críticas</td></tr>';

  const f2v=CLASSES.map(c=>m.f2_by_class[c]||0);
  const fnrv=CLASSES.map(c=>m.fn_rate_by_class[c]||0);
  const wv=CLASSES.map(c=>m.weight_by_class[c]||0);
  const barC=f2v.map((v,i)=>{{
    const isCrit=wv[i]>0;
    if(v>=0.9) return isCrit?'rgba(34,211,165,0.9)':'rgba(34,211,165,0.45)';
    if(v>=0.7) return isCrit?'rgba(94,234,212,0.85)':'rgba(94,234,212,0.4)';
    if(v>=0.5) return isCrit?'rgba(124,106,247,0.85)':'rgba(124,106,247,0.4)';
    if(v>0)    return isCrit?'rgba(245,158,11,0.85)':'rgba(245,158,11,0.4)';
    return isCrit?'rgba(239,68,68,0.85)':'rgba(239,68,68,0.35)';
  }});
  if(classChartInst) classChartInst.destroy();
  classChartInst=new Chart(document.getElementById('classChart').getContext('2d'),{{
    type:'bar',data:{{labels:CLS_LBL,datasets:[
      {{label:'F2 Score',data:f2v,backgroundColor:barC,borderRadius:5,borderSkipped:false,order:2,yAxisID:'y'}},
      {{label:'FN Rate',data:fnrv,type:'line',borderColor:'rgba(239,68,68,0.85)',
        backgroundColor:'rgba(239,68,68,0.08)',fill:true,tension:0.3,
        pointRadius:5,pointBackgroundColor:'rgba(239,68,68,0.9)',borderWidth:2,order:1,yAxisID:'y'}}
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#8892b0',font:{{size:11}},boxWidth:12}}}},
        tooltip:{{callbacks:{{label:c=>` ${{c.dataset.label}}: ${{c.parsed.y.toFixed(3)}}`}}}}}},
      scales:{{
        x:{{grid:{{color:'rgba(255,255,255,.04)'}},ticks:{{color:'#8892b0',font:{{size:10}}}}}},
        y:{{min:0,max:1,grid:{{color:'rgba(255,255,255,.06)'}},
            ticks:{{color:'#8892b0',callback:v=>v.toFixed(1)}}}}
      }}
    }}
  }});
  document.getElementById('detail-panel').scrollIntoView({{behavior:'smooth',block:'start'}});
}}

// ── SCATTER ───────────────────────────────────────────────────────────────────
new Chart(document.getElementById('scatterChart').getContext('2d'),{{
  type:'scatter',
  data:{{datasets:[{{
    label:'Modelos',
    data:MODELS.map(m=>({{x:m.score_100,y:m.r_final_100,name:m.short}})),
    backgroundColor:MODELS.map(m=>P_COLORS[m.paradigm]||'rgba(200,200,200,0.7)'),
    pointRadius:14,pointHoverRadius:17
  }}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:c=>`${{c.raw.name}}  Score: ${{c.raw.x.toFixed(1)}}  R: ${{c.raw.y.toFixed(1)}}%`}}}}
    }},
    scales:{{
      x:{{min:0,max:105,title:{{display:true,text:'Score (/100)',color:'#8892b0',font:{{size:12}}}},
          grid:{{color:'rgba(255,255,255,.06)'}},ticks:{{color:'#8892b0'}}}},
      y:{{min:0,max:105,title:{{display:true,text:'R_final (%)',color:'#8892b0',font:{{size:12}}}},
          grid:{{color:'rgba(255,255,255,.06)'}},ticks:{{color:'#8892b0'}}}}
    }}
  }}
}});

// ── F2 COMPARISON ─────────────────────────────────────────────────────────────
new Chart(document.getElementById('compF2Chart').getContext('2d'),{{
  type:'bar',
  data:{{labels:CLS_LBL,datasets:MODELS.map(m=>{{
    return {{label:m.short,data:CLASSES.map(c=>m.f2_by_class[c]||0),
             backgroundColor:P_COLORS[m.paradigm]||'rgba(200,200,200,0.7)',
             borderRadius:3,borderSkipped:false}};
  }})}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:'#8892b0',font:{{size:10}},boxWidth:12,padding:10}}}},
      tooltip:{{callbacks:{{label:c=>` ${{c.dataset.label}}: ${{c.parsed.y.toFixed(3)}}`}}}}}},
    scales:{{
      x:{{grid:{{color:'rgba(255,255,255,.04)'}},ticks:{{color:'#8892b0',font:{{size:10}}}}}},
      y:{{min:0,max:1,grid:{{color:'rgba(255,255,255,.06)'}},ticks:{{color:'#8892b0',callback:v=>v.toFixed(1)}}}}
    }}
  }}
}});

// ── FN RATE COMPARISON ────────────────────────────────────────────────────────
new Chart(document.getElementById('compFNChart').getContext('2d'),{{
  type:'bar',
  data:{{labels:CLS_LBL,datasets:MODELS.map(m=>{{
    const c=P_COLORS[m.paradigm]||'rgba(200,200,200,0.6)';
    return {{label:m.short,data:CLASSES.map(cl=>m.fn_rate_by_class[cl]||0),
             backgroundColor:c.replace(/[\d.]+\)$/,'0.65)'),
             borderRadius:3,borderSkipped:false}};
  }})}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:'#8892b0',font:{{size:10}},boxWidth:12,padding:10}}}},
      tooltip:{{callbacks:{{label:c=>` ${{c.dataset.label}}: ${{(c.parsed.y*100).toFixed(0)}}% FN`}}}}}},
    scales:{{
      x:{{grid:{{color:'rgba(255,255,255,.04)'}},ticks:{{color:'#8892b0',font:{{size:10}}}}}},
      y:{{min:0,max:1,grid:{{color:'rgba(255,255,255,.06)'}},
          ticks:{{color:'#8892b0',callback:v=>(v*100).toFixed(0)+'%'}}}}
    }}
  }}
}});

showDetail(0);
</script>
</body>
</html>"""
    return html


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Panel HTML v3 — Privacy Detection Score (sin InfraScore)"
    )
    parser.add_argument("input", help="JSON generado por Analisis_Score_Chatbot_v3.py")
    parser.add_argument("-o","--output", default=None, help="Archivo HTML de salida")
    parser.add_argument("--open", action="store_true", help="Abrir en navegador")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌  Archivo no encontrado: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"📂  Leyendo: {args.input}")
    data    = load_json(args.input)
    results = data if isinstance(data, list) else data.get("results", [])

    if not results:
        print("❌  No se encontraron resultados.", file=sys.stderr)
        sys.exit(1)

    models = extract_models(results)
    print(f"✅  {len(models)} modelo(s) cargados\n")
    print(f"{'#':<3} {'Modelo':<42} {'Paradigma':<22} {'Score':>6}  {'R_final':>8}  {'R1':>6}")
    print("─" * 88)
    for m in models:
        pm = pmeta(m["paradigm"])
        print(f"#{m['rank']:<2} {m['short']:<42} {pm['icon']} {pm['label']:<20} "
              f"{m['score_100']:>6.2f}  {m['r_final_100']:>7.1f}%  {m['r1_100']:>5.1f}%")

    html = build_html(models, args.input)
    out  = args.output or (os.path.splitext(args.input)[0] + "_panel_v3.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n🎉  Panel → {out}")
    if args.open:
        webbrowser.open(f"file://{os.path.abspath(out)}")
        print("🌐  Abriendo en el navegador…")


if __name__ == "__main__":
    main()
