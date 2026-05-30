"""Inved Corp AI — PoC d'estimation immobilière (NiceGUI). Dark + gold luxury theme.

Slim dark rail (brand + Estimation + Monitoring) + dark glass content, gold accents.
Live estimate with a casino count-up price + animated charts (ECharts).

Run (racine du dépôt) :  python app/app.py   ->  http://localhost:8502
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nicegui import ui

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import get_client                 # noqa: E402
from features import build_row, engineer_features, load_defaults  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# ---- loaded once at startup ----
CLIENT = get_client()
DEFAULTS = load_defaults()
RMSLE = 0.1122
KITCHEN = {"Po": "Médiocre", "Fa": "Passable", "TA": "Moyenne", "Gd": "Bonne", "Ex": "Excellente"}

_train = pd.read_csv(ROOT / "data" / "train.csv")
PRICES = np.sort(_train["SalePrice"].to_numpy())
NEIGH_MED = _train.groupby("Neighborhood")["SalePrice"].median().sort_values()
MARKET = {
    "OverallQual": float(_train["OverallQual"].median()),
    "OverallCond": float(_train["OverallCond"].median()),
    "GrLivArea": float(_train["GrLivArea"].median()),
    "YearBuilt": float(_train["YearBuilt"].median()),
    "GarageCars": float(_train["GarageCars"].median()),
}

# ---- monitoring data + helpers ----
import json as _json  # noqa: E402

_eq_path = Path(__file__).resolve().parent / "monitoring_data.json"
EQUITY = _json.loads(_eq_path.read_text())["equity"] if _eq_path.exists() else {}
TRAIN_FEAT = engineer_features(_train.copy())
TEST_FEAT = engineer_features(pd.read_csv(ROOT / "data" / "test.csv"))
PSI_FEATS = ["GrLivArea", "TotalSF", "OverallQual", "1stFlrSF", "LotArea",
             "HouseAge", "GarageCars", "TotalBathrooms", "YearBuilt"]


def psi(expected, actual, bins=10):
    cuts = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return float("nan")
    e = np.histogram(expected.dropna(), bins=cuts)[0] / max(expected.notna().sum(), 1)
    a = np.histogram(actual.dropna(), bins=cuts)[0] / max(actual.notna().sum(), 1)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_psi(test_df):
    return {f: psi(TRAIN_FEAT[f], test_df[f]) for f in PSI_FEATS if f in TRAIN_FEAT and f in test_df}


def gentrified_test():
    g = TEST_FEAT.copy()
    m = g["Neighborhood"] == "NAmes"  # one quartier gentrifies
    g.loc[m, "OverallQual"] = (g.loc[m, "OverallQual"] + 3).clip(upper=10)
    g.loc[m, "TotalSF"] = g.loc[m, "TotalSF"] * 1.30
    g.loc[m, "YearsSinceRemodel"] = (g.loc[m, "YearsSinceRemodel"] - 25).clip(lower=0)
    g.loc[m, "HouseAge"] = (g.loc[m, "HouseAge"] - 20).clip(lower=0)
    return g


def psi_band(v):
    if v > 0.25:
        return "dérive forte", "#d39090"
    if v > 0.1:
        return "à surveiller", GOLD_SOFT
    return "stable", "#6fd38d"


def psi_chart_opt(psi_dict):
    feats = list(psi_dict.keys())
    bars = [{"value": round(psi_dict[f], 3), "itemStyle": {"color": psi_band(psi_dict[f])[1]}} for f in feats]
    o = echart_base()
    o.update({
        "grid": {"left": 48, "right": 18, "top": 24, "bottom": 66},
        "xAxis": {"type": "category", "data": feats, "axisLabel": {"rotate": 40, "color": AXIS, "fontSize": 10},
                  "axisLine": {"lineStyle": {"color": "rgba(255,255,255,.15)"}}},
        "yAxis": {"type": "value", "axisLabel": {"color": AXIS},
                  "splitLine": {"lineStyle": {"color": "rgba(255,255,255,.06)"}}},
        "series": [{"type": "bar", "data": bars, "barWidth": "55%",
                    "markLine": {"silent": True, "symbol": "none", "data": [
                        {"yAxis": 0.1, "lineStyle": {"color": GOLD_SOFT, "type": "dashed"}, "label": {"formatter": "0,1", "color": AXIS}},
                        {"yAxis": 0.25, "lineStyle": {"color": "#d39090", "type": "dashed"}, "label": {"formatter": "0,25", "color": AXIS}}]}}],
    })
    return o


def rmsle_gauge_opt():
    return {"backgroundColor": "transparent", "series": [{
        "type": "gauge", "min": 0.08, "max": 0.18, "radius": "95%", "center": ["50%", "58%"],
        "axisLine": {"lineStyle": {"width": 14, "color": [[0.5, GOLD], [1, "rgba(211,144,144,.65)"]]}},
        "pointer": {"itemStyle": {"color": GOLD_SOFT}}, "progress": {"show": False},
        "axisLabel": {"color": AXIS, "fontSize": 9, "distance": -40}, "axisTick": {"show": False},
        "splitLine": {"length": 10, "lineStyle": {"color": "rgba(255,255,255,.25)"}}, "title": {"show": False},
        "detail": {"formatter": "{value}", "color": INK, "fontSize": 22, "offsetCenter": [0, "64%"]},
        "data": [{"value": 0.1122}]}]}


def deployment_rows():
    try:
        import mlflow as _m
        from mlflow.tracking import MlflowClient
        _m.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
        c = MlflowClient()
        out = []
        for mv in c.search_model_versions("name='inved-house-price'"):
            out.append({"version": str(mv.version), "alias": ",".join(getattr(mv, "aliases", []) or []) or "—",
                        "run_id": (mv.run_id or "")[:8], "statut": mv.status or "—"})
        return sorted(out, key=lambda r: int(r["version"]))
    except Exception:
        return []


def equity_box_opt():
    names = list(EQUITY.keys())
    boxdata = []
    for n in names:
        med = EQUITY[n]["median"]
        col = "#d39090" if abs(med) > 8 else GOLD
        boxdata.append({"value": EQUITY[n]["box"], "itemStyle": {"color": "rgba(212,175,55,.16)", "borderColor": col}})
    o = echart_base()
    o.update({
        "title": {"text": "Équité — résidus par quartier (%, holdout)", "left": "center",
                  "textStyle": {"color": INK, "fontSize": 13, "fontWeight": 600}},
        "grid": {"left": 52, "right": 20, "top": 40, "bottom": 80},
        "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 45, "color": AXIS, "fontSize": 9},
                  "axisLine": {"lineStyle": {"color": "rgba(255,255,255,.15)"}}},
        "yAxis": {"type": "value", "axisLabel": {"color": AXIS, "formatter": "{value}%"},
                  "splitLine": {"lineStyle": {"color": "rgba(255,255,255,.06)"}}},
        "series": [{"type": "boxplot", "data": boxdata}],
    })
    return o

# ---- palette ----
BG = "#0d0f13"
INK = "#e8eaed"
MUTE = "#8b929b"
GOLD = "#d4af37"
GOLD_SOFT = "#e8c766"
PANEL = "rgba(255,255,255,.04)"
LINE = "rgba(212,175,55,.22)"
AXIS = "#c9d0d8"  # chart axis/label text: bright but not pure white

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/countup.js/2.8.0/countUp.umd.js"></script>
<style>
  :root { --gold:#d4af37; }
  body { background: radial-gradient(1200px 600px at 80% -10%, rgba(212,175,55,.10), transparent 60%), #0d0f13 !important; color:#e8eaed; font-family:'Inter',sans-serif; }
  .q-page, .nicegui-content { background: transparent !important; }
  .serif { font-family:'Playfair Display', serif; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
  .glass { background:rgba(255,255,255,.04); border:1px solid rgba(212,175,55,.22); border-radius:16px;
           backdrop-filter:blur(10px); box-shadow:0 8px 30px rgba(0,0,0,.35); animation:fadeUp .5s ease both;
           transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
  .glass:hover { transform:translateY(-3px); box-shadow:0 14px 40px rgba(0,0,0,.5); border-color:rgba(212,175,55,.5); }
  .navitem { transition:background .2s ease, border-color .2s ease; border-left:3px solid transparent; }
  .navitem:hover { background:rgba(212,175,55,.08); }
  .navactive { background:linear-gradient(90deg, rgba(212,175,55,.18), transparent); border-left:3px solid var(--gold);
               box-shadow:inset 0 0 24px rgba(212,175,55,.12); }
  .q-field__native, .q-field__label, input { color:#e8eaed !important; }
  .q-field__control:before { border-color:rgba(212,175,55,.3) !important; }
  ::-webkit-scrollbar{width:9px;height:9px} ::-webkit-scrollbar-thumb{background:rgba(212,175,55,.3);border-radius:8px}
  /* dropdown popups: dark + gold hover (fix white-on-white) */
  .q-menu{background:#15171c !important;color:#e8eaed !important;border:1px solid rgba(212,175,55,.28);box-shadow:0 12px 34px rgba(0,0,0,.6)}
  .q-item,.q-item__label,.q-item__section{color:#e8eaed !important}
  .q-item:hover,.q-item--active,.q-manual-focusable--focused{background:rgba(212,175,55,.18) !important;color:#fff !important}
  /* remove dated native number spinners */
  input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none;margin:0}
  input[type=number]{-moz-appearance:textfield}
</style>
"""


def echart_base():
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": AXIS, "fontFamily": "Inter"},
        "grid": {"left": 48, "right": 18, "top": 36, "bottom": 30},
        "animationDuration": 900,
        "animationEasing": "cubicOut",
    }


def market_chart_opt(price):
    counts, edges = np.histogram(PRICES, bins=28)
    centers = ((edges[:-1] + edges[1:]) / 2).astype(int)
    hit = int(np.clip(np.searchsorted(edges, price) - 1, 0, len(counts) - 1))
    pct = float((PRICES < price).mean() * 100)
    bars = [{"value": int(c), "itemStyle": {"color": GOLD if i == hit else "rgba(232,234,237,.18)"}}
            for i, c in enumerate(counts)]
    o = echart_base()
    o.update({
        "title": {"text": f"Top {100 - pct:.0f}% du marché d'Ames", "left": "center",
                  "textStyle": {"color": INK, "fontSize": 13, "fontWeight": 600}},
        "xAxis": {"type": "category", "data": [f"{v//1000}k" for v in centers],
                  "axisLine": {"lineStyle": {"color": "rgba(255,255,255,.15)"}},
                  "axisLabel": {"interval": 4, "color": AXIS}},
        "yAxis": {"type": "value", "axisLabel": {"color": AXIS},
                  "splitLine": {"lineStyle": {"color": "rgba(255,255,255,.06)"}}},
        "series": [{"type": "bar", "data": bars, "barWidth": "85%"}],
    })
    return o


def comps_chart_opt(neigh, price):
    chosen = NEIGH_MED.tail(13)
    if neigh in NEIGH_MED.index and neigh not in chosen.index:  # always include the selected quartier
        chosen = pd.concat([chosen, NEIGH_MED.loc[[neigh]]]).sort_values()
    cats = list(chosen.index)
    data = [{"value": int(v), "itemStyle": {"color": GOLD if n == neigh else "rgba(232,234,237,.18)"}}
            for n, v in chosen.items()]
    o = echart_base()
    o.update({
        "grid": {"left": 72, "right": 26, "top": 36, "bottom": 54},
        "title": {"text": "Prix médian par quartier", "left": "center",
                  "textStyle": {"color": INK, "fontSize": 13, "fontWeight": 600}},
        "xAxis": {"type": "category", "data": cats,
                  "axisLabel": {"rotate": 40, "fontSize": 11, "color": AXIS, "fontWeight": 500},
                  "axisLine": {"lineStyle": {"color": "rgba(255,255,255,.15)"}}},
        "yAxis": {"type": "value", "axisLabel": {"color": AXIS},
                  "splitLine": {"lineStyle": {"color": "rgba(255,255,255,.06)"}}},
        "series": [{"type": "bar", "data": data, "barWidth": "60%",
                    "markLine": {"silent": True, "symbol": "none",
                                 "lineStyle": {"color": "#ffffff", "type": "dashed", "width": 2},
                                 "label": {"formatter": "Estimation IA", "position": "insideStartTop",
                                           "color": "#ffffff", "fontWeight": 700},
                                 "data": [{"yAxis": int(price)}]}}],
    })
    return o


def radar_chart_opt(form):
    def pctl(arr_col, v):
        return float((_train[arr_col] < v).mean() * 100)
    bien = [form["OverallQual"] * 10, form["OverallCond"] * 10, pctl("GrLivArea", form["GrLivArea"]),
            pctl("YearBuilt", form["YearBuilt"]), form["GarageCars"] / 4 * 100]
    mkt = [MARKET["OverallQual"] * 10, MARKET["OverallCond"] * 10, pctl("GrLivArea", MARKET["GrLivArea"]),
           pctl("YearBuilt", MARKET["YearBuilt"]), MARKET["GarageCars"] / 4 * 100]
    o = echart_base()
    o.update({
        "title": {"text": "Profil du bien vs marché", "left": "center",
                  "textStyle": {"color": INK, "fontSize": 13, "fontWeight": 600}},
        "legend": {"bottom": 0, "textStyle": {"color": AXIS}, "data": ["Ce bien", "Marché médian"]},
        "radar": {"indicator": [{"name": n, "max": 100} for n in ["Qualité", "État", "Surface", "Récence", "Garage"]],
                  "axisName": {"color": AXIS}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,.08)"}},
                  "splitArea": {"areaStyle": {"color": ["rgba(212,175,55,.03)", "transparent"]}}},
        "series": [{"type": "radar", "data": [
            {"value": [round(x) for x in bien], "name": "Ce bien",
             "itemStyle": {"color": GOLD}, "areaStyle": {"color": "rgba(212,175,55,.35)"}},
            {"value": [round(x) for x in mkt], "name": "Marché médian",
             "itemStyle": {"color": "rgba(232,234,237,.5)"}, "lineStyle": {"type": "dashed"}}]}],
    })
    return o


FEATURE_FR = {
    "TotalSF": "Surface totale", "OverallQual": "Qualité globale", "Neighborhood": "Quartier",
    "GarageCars": "Garage (places)", "TotalBathrooms": "Salles de bain", "GrLivArea": "Surface habitable",
    "KitchenQual": "Qualité cuisine", "YearBuilt": "Année de construction", "1stFlrSF": "Surface 1er niveau",
    "LotArea": "Terrain", "OverallCond": "État général", "BsmtUnfSF": "Sous-sol non fini",
    "FullBath": "Salles de bain", "HouseAge": "Âge du bien", "YearsSinceRemodel": "Réno. (années)",
    "GarageAge": "Âge du garage", "2ndFlrSF": "Surface étage", "ExterQual": "Qualité extérieure",
    "BsmtFinSF1": "Sous-sol fini", "MSZoning": "Zonage", "Fireplaces": "Cheminées", "CentralAir": "Climatisation",
}


def _xgb_parts():
    """XGBoost base of the Stacking champion (preprocessor, model) for SHAP — loaded once."""
    obj = getattr(CLIENT, "model", None)
    if obj is None:
        import mlflow as _m
        _m.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
        obj = _m.sklearn.load_model("models:/inved-house-price@Production")
    try:
        p = obj.named_estimators_["xgb"]
        return p[:-1], p[-1]
    except Exception:
        return None, None


SHAP_PREP, SHAP_MODEL = _xgb_parts()


def shap_top_factors(row, k=6):
    """Per-estimate factors via XGBoost-native SHAP (pred_contribs). Returns [(label, %effect)]."""
    if SHAP_MODEL is None:
        return []
    import xgboost
    Xt = SHAP_PREP.transform(row)
    cols = list(Xt.columns) if hasattr(Xt, "columns") else list(SHAP_PREP.get_feature_names_out())
    contribs = SHAP_MODEL.get_booster().predict(
        xgboost.DMatrix(Xt, enable_categorical=True), pred_contribs=True)[0][:-1]
    pairs = sorted(zip(cols, contribs), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return [(FEATURE_FR.get(c, c), float((np.exp(v) - 1) * 100)) for c, v in pairs]


def shap_chart_opt(factors):
    factors = list(reversed(factors))  # ECharts horizontal bars render bottom-up
    bars = [{"value": round(f[1], 1),
             "itemStyle": {"color": GOLD if f[1] >= 0 else "rgba(232,234,237,.30)"}} for f in factors]
    o = echart_base()
    o.update({
        "grid": {"left": 150, "right": 60, "top": 36, "bottom": 28},
        "title": {"text": "Facteurs de l'estimation (IA)", "left": "center",
                  "textStyle": {"color": INK, "fontSize": 13, "fontWeight": 600}},
        "xAxis": {"type": "value", "axisLabel": {"color": AXIS, "formatter": "{value}%"},
                  "splitLine": {"lineStyle": {"color": "rgba(255,255,255,.06)"}}},
        "yAxis": {"type": "category", "data": [f[0] for f in factors], "axisLabel": {"color": AXIS},
                  "axisLine": {"lineStyle": {"color": "rgba(255,255,255,.15)"}}},
        "series": [{"type": "bar", "data": bars, "barWidth": "62%",
                    "label": {"show": True, "position": "right", "formatter": "{c}%", "color": AXIS, "fontSize": 10}}],
    })
    return o


def nav_item(label, icon, path, active):
    cls = "navitem navactive" if active else "navitem"
    row = ui.row().classes(f"{cls} items-center gap-3 w-full cursor-pointer").style("padding:13px 20px")
    with row:
        ui.icon(icon).style(f"color:{GOLD if active else 'rgba(255,255,255,.6)'}")
        ui.label(label).style(f"color:{'#fff;font-weight:600' if active else 'rgba(255,255,255,.75)'}")
    row.on("click", lambda: ui.navigate.to(path))


def left_rail(active):
    with ui.left_drawer(fixed=True, bordered=False).style(f"background:#0a0b0d;padding:0").classes("w-64"):
        with ui.row().classes("items-center gap-3").style("padding:24px 20px 18px"):
            ui.icon("diamond", size="26px").style(f"color:{GOLD}")
            with ui.column().classes("gap-0"):
                ui.label("INVED CORP AI").classes("serif").style(f"color:{GOLD};font-weight:800;font-size:1.1rem;letter-spacing:1px;line-height:1")
                ui.label("Estimation immobilière de prestige").style(f"color:{MUTE};font-size:.68rem;margin-top:3px")
        ui.separator().style(f"background:{LINE}")
        nav_item("Estimation", "real_estate_agent", "/", active == "Estimation")
        nav_item("Monitoring", "monitoring", "/monitoring", active == "Monitoring")
        ui.space()
        chip = "API MLflow :5000" if CLIENT.mode == "mlflow-api" else "modèle en mémoire"
        with ui.row().classes("items-center gap-2").style("padding:16px 20px"):
            ui.icon("circle", size="9px").style("color:#6fd38d")
            ui.label(f"Inférence · {chip}").style(f"color:{MUTE};font-size:.7rem")


def page_chrome(active, title):
    ui.dark_mode().enable()  # Quasar dark variants -> dropdowns/menus render dark (no white-on-white)
    ui.colors(primary=GOLD)
    ui.add_head_html(HEAD)
    left_rail(active)
    with ui.column().classes("gap-1").style("margin:8px 4px 16px"):
        ui.label(title).classes("serif").style(f"color:{INK};font-size:2rem;font-weight:700")
        ui.element("div").style(f"width:60px;height:3px;background:linear-gradient(90deg,{GOLD},transparent);border-radius:3px")


@ui.page("/")
def estimation_page():
    page_chrome("Estimation", "Estimation immobilière")
    rd = DEFAULTS["raw_defaults"]
    lbl = lambda t: ui.label(t).style(f"color:{GOLD};font-weight:600;font-size:.78rem;letter-spacing:.5px;text-transform:uppercase")

    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with ui.element("div").classes("glass").style("flex:1;padding:24px"):
            with ui.grid(columns=3).classes("w-full gap-x-8 gap-y-3"):
                with ui.column().classes("gap-3"):
                    lbl("Surface & terrain")
                    gr = ui.number("Surface habitable (sq ft)", value=int(rd["GrLivArea"]), min=300, max=6000, step=50).props("dense").classes("w-full")
                    flr = ui.number("Surface 1er niveau (sq ft)", value=int(rd["1stFlrSF"]), min=300, max=4000, step=50).props("dense").classes("w-full")
                    lot = ui.number("Terrain (sq ft)", value=int(rd["LotArea"]), min=1000, max=60000, step=500).props("dense").classes("w-full")
                with ui.column().classes("gap-1"):
                    lbl("Qualité & état")
                    ui.label("Qualité globale").style(f"color:{MUTE};font-size:.8rem;margin-top:6px")
                    oq = ui.slider(min=1, max=10, value=int(rd["OverallQual"])).props("label")
                    ui.label("État général").style(f"color:{MUTE};font-size:.8rem")
                    oc = ui.slider(min=1, max=10, value=int(rd["OverallCond"])).props("label")
                    kq = ui.select(KITCHEN, value=rd.get("KitchenQual", "TA"), label="Qualité cuisine").props("dense").classes("w-full")
                with ui.column().classes("gap-1"):
                    lbl("Localisation & annexes")
                    neigh = ui.select(DEFAULTS["categorical_options"]["Neighborhood"], value=rd["Neighborhood"],
                                      label="Quartier", with_input=True).props("dense").classes("w-full")
                    yb = ui.number("Année de construction", value=int(rd["YearBuilt"]), min=1872, max=2010).props("dense").classes("w-full")
                    ui.label("Places de garage").style(f"color:{MUTE};font-size:.8rem;margin-top:6px")
                    gc = ui.slider(min=0, max=4, value=int(rd["GarageCars"])).props("label")
                    ui.label("Salles de bain").style(f"color:{MUTE};font-size:.8rem")
                    fb = ui.slider(min=0, max=4, value=int(rd["FullBath"])).props("label")

        with ui.element("div").classes("glass").style("width:330px;padding:26px;text-align:center"):
            ui.label("PRIX ESTIMÉ").style(f"color:{MUTE};font-size:.75rem;letter-spacing:2px")
            ui.html('<div id="inved-price" class="serif" style="color:#d4af37;font-size:3rem;font-weight:800;line-height:1.15;margin:6px 0">—</div>')
            band_lbl = ui.label("").style(f"color:{MUTE};font-size:.82rem")
            ui.element("div").style(f"height:1px;background:{LINE};margin:18px 0")
            ui.label("PRINCIPAUX FACTEURS").style(f"color:{GOLD};font-size:.72rem;letter-spacing:1px")
            factors_box = ui.column().classes("gap-1 w-full items-start").style("margin-top:6px")
            meta_lbl = ui.label("").style(f"color:#5a626c;font-size:.68rem;margin-top:14px")

    # charts row
    with ui.grid(columns=2).classes("w-full gap-5").style("margin-top:20px"):
        c_market = ui.echart(market_chart_opt(150000)).classes("glass").style("height:300px;padding:8px")
        c_radar = ui.echart(radar_chart_opt({"OverallQual": 6, "OverallCond": 5, "GrLivArea": 1464, "YearBuilt": 1973, "GarageCars": 2})).classes("glass").style("height:300px;padding:8px")
    with ui.element("div").classes("glass w-full").style("margin-top:20px;padding:8px"):
        c_shap = ui.echart(shap_chart_opt([])).style("height:300px;width:100%")
    with ui.element("div").classes("glass w-full").style("margin-top:20px;padding:8px"):
        c_comps = ui.echart(comps_chart_opt("NAmes", 150000)).style("height:320px;width:100%")

    def update():
        try:
            form = {"GrLivArea": gr.value, "1stFlrSF": flr.value, "LotArea": lot.value,
                    "OverallQual": int(oq.value), "OverallCond": int(oc.value), "KitchenQual": kq.value,
                    "Neighborhood": neigh.value, "YearBuilt": yb.value, "GarageCars": int(gc.value), "FullBath": int(fb.value)}
            row = build_row(form, DEFAULTS)
            pred_log, latency = CLIENT.predict(row)
            price = float(np.expm1(pred_log))
            lo, hi = price * np.exp(-RMSLE), price * np.exp(RMSLE)
            ui.run_javascript(
                f"if(window.countUp){{if(!window._ip){{window._ip=new countUp.CountUp('inved-price',{price},"
                f"{{duration:1.2,separator:',',prefix:'$'}});window._ip.start();}}else{{window._ip.update({price});}}}}"
                f"else{{document.getElementById('inved-price').textContent='${price:,.0f}';}}"
            )
            band_lbl.set_text(f"Fourchette indicative : ${lo:,.0f} — ${hi:,.0f}")
            meta_lbl.set_text(f"{CLIENT.mode} · {latency*1000:.0f} ms · fourchette non calibrée")
            facts = shap_top_factors(row, k=6)
            for ch, opt in ((c_market, market_chart_opt(price)),
                            (c_comps, comps_chart_opt(neigh.value, price)),
                            (c_radar, radar_chart_opt(form)),
                            (c_shap, shap_chart_opt(facts))):
                ch.options.clear()
                ch.options.update(opt)
                ch.update()
            factors_box.clear()
            with factors_box:
                if facts:
                    for name, pct in facts[:3]:
                        col = GOLD if pct >= 0 else "#d39090"
                        ui.label(f"{'▲' if pct >= 0 else '▼'} {name}   {pct:+.0f}%").style(
                            f"color:{col};font-size:.84rem;font-weight:500")
                else:
                    ui.label("facteurs indisponibles").style(f"color:{MUTE};font-size:.78rem")
        except Exception as exc:
            ui.notify(f"Estimation impossible : {exc}", type="negative")

    for w in (gr, flr, lot, oq, oc, kq, neigh, yb, gc, fb):
        w.on_value_change(lambda: update())
    update()


@ui.page("/monitoring")
def monitoring_page():
    page_chrome("Monitoring", "Supervision du modèle")
    with ui.row().classes("items-center justify-between w-full").style("margin-bottom:4px"):
        ui.label("Vue équipe data science / ops — les 3 axes d'évaluation de la Phase 5, en continu.").style(f"color:{MUTE}")
        ui.link("Ouvrir MLflow UI ↗", "http://127.0.0.1:5001", new_tab=True).style(
            f"color:{GOLD};text-decoration:none;font-weight:600")

    # ===== Couche 1 — mathématique =====
    ui.label("Couche 1 — Mathématique").classes("serif").style(f"color:{GOLD};font-size:1.2rem;margin-top:12px")
    state = {"g": False}
    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with ui.element("div").classes("glass").style("flex:2;padding:18px"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Dérive des données (PSI) — train vs test Kaggle").style(f"color:{INK};font-weight:600")
                sw = ui.switch("Simuler une gentrification").props("color=amber")
            psi_chart = ui.echart(psi_chart_opt(compute_psi(TEST_FEAT))).style("height:250px;width:100%")
            psi_status = ui.label("").style(f"color:{MUTE};font-size:.82rem;margin-top:4px")
        with ui.element("div").classes("glass").style("flex:1;padding:18px;text-align:center"):
            ui.label("RMSLE réalisé").style(f"color:{INK};font-weight:600")
            ui.echart(rmsle_gauge_opt()).style("height:190px;width:100%")
            ui.label("seuil de réentraînement : 0,13").style(f"color:{MUTE};font-size:.76rem")

    def refresh_psi():
        ps = compute_psi(gentrified_test() if state["g"] else TEST_FEAT)
        psi_chart.options.clear(); psi_chart.options.update(psi_chart_opt(ps)); psi_chart.update()
        mx = max(ps.values()); band, _ = psi_band(mx)
        psi_status.set_text(f"PSI max = {mx:.3f} → {band}" + ("   ·   réentraînement déclenché" if mx > 0.25 else ""))

    sw.on_value_change(lambda e: (state.update(g=e.value), refresh_psi()))
    refresh_psi()

    with ui.row().classes("w-full gap-4").style("margin-top:8px"):
        for txt in ["Cadence mensuelle (fenêtre glissante)", "RMSLE ≤ 0,13", "PSI < 0,25"]:
            with ui.element("div").classes("glass").style("flex:1;padding:12px"):
                ui.label("● " + txt).style("color:#6fd38d;font-size:.82rem")

    # ===== Couche 2 — système =====
    ui.label("Couche 2 — Système").classes("serif").style(f"color:{GOLD};font-size:1.2rem;margin-top:22px")
    import time as _t
    _row = build_row({}, DEFAULTS)
    _lat = []
    for _ in range(15):
        _t0 = _t.perf_counter(); CLIENT.predict(_row); _lat.append((_t.perf_counter() - _t0) * 1000)
    p50, p95 = float(np.percentile(_lat, 50)), float(np.percentile(_lat, 95))
    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with ui.element("div").classes("glass").style("flex:1;padding:20px"):
            ui.label("Latence d'inférence").style(f"color:{INK};font-weight:600")
            ui.label(f"p50 {p50:.0f} ms · p95 {p95:.0f} ms").style(f"color:{GOLD};font-size:1.5rem;font-weight:700;margin:6px 0")
            ui.label("SLA Canvas < 5 000 ms — marge ~90× (proxy in-process)").style(f"color:{MUTE};font-size:.76rem")
        with ui.element("div").classes("glass").style("flex:2;padding:20px"):
            ui.label("Versions déployées — MLflow Registry").style(f"color:{INK};font-weight:600;margin-bottom:6px")
            rows = deployment_rows()
            if rows:
                ui.table(columns=[{"name": k, "label": k.capitalize(), "field": k} for k in rows[0]],
                         rows=rows).props("dense flat").style("background:transparent;color:#e8eaed")
            else:
                ui.label("registre indisponible — lancer `mlflow ui`").style(f"color:{MUTE};font-size:.8rem")
    ui.label("Aussi surveillés : taux d'erreur 5xx (seuil 1 %), volume de requêtes, logs de prédiction "
             "(rétention 90 j pour l'audit d'équité), événements de déploiement.").style(f"color:{MUTE};font-size:.76rem;margin-top:6px")

    # ===== Couche 3 — métier =====
    ui.label("Couche 3 — Métier").classes("serif").style(f"color:{GOLD};font-size:1.2rem;margin-top:22px")
    with ui.row().classes("w-full gap-4"):
        for big, sub, lab in [("4 h → 2 h", "−50 % temps d'expertise", "Temps d'expertise"),
                              ("+20 %", "ventes sous 90 jours", "Conversion 90 j"),
                              ("< 10 %", "décisions corrigées", "Taux d'override")]:
            with ui.element("div").classes("glass").style("flex:1;padding:18px"):
                ui.label(lab).style(f"color:{MUTE};font-size:.74rem;letter-spacing:1px")
                ui.label(big).classes("serif").style(f"color:{GOLD};font-size:1.7rem;font-weight:700")
                ui.label(sub).style(f"color:{MUTE};font-size:.74rem")
                ui.label("valeur illustrative — mesurée après déploiement").style("color:#5a626c;font-size:.66rem;margin-top:6px")
    with ui.element("div").classes("glass w-full").style("margin-top:16px;padding:8px"):
        ui.echart(equity_box_opt()).style("height:340px;width:100%")
    ui.label("Audit d'équité (anti-redlining) : médianes de résidus par quartier surveillées en continu ; léger "
             "penchant à sous-estimer BrkSide / IDOTRR à suivre (cf. NB4 §5.5.4). Rollout A/B 50/50 sur 3 mois avant généralisation.").style(
        f"color:{MUTE};font-size:.78rem;margin-top:8px")


ui.run(host="127.0.0.1", port=8502, reload=False, show=False, title="Inved Corp AI",
       favicon=str(Path(__file__).resolve().parent / "favicon.png"))
