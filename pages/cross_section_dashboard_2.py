import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# Optional for LOWESS
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False

# ----------------------- Look & feel -----------------------
color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]
st.set_page_config(page_title='Scatter — Firms (HU cross-section, simulated)', layout='wide')

# ----------------------- Loaders ---------------------------
@st.cache_data
def load_cross_section(path: str = 'data/synthetic/sim_cs2019_by_nace2_withcats.parquet') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {p}")
        st.stop()
    df = pd.read_parquet(p).copy()
    # ensure a clean 2-digit NACE code
    if "nace2" in df.columns:
        df["nace2_2d"] = (
            df["nace2"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna("")
            .str.zfill(2)
            .str[:2]
        )
    else:
        st.error("Column `nace2` not found in the simulated data.")
        st.stop()
    return df

@st.cache_data
def load_nace2_labels(path: str = "nace2_labels.xlsx") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_excel(p, dtype=str)
    if "nace2" not in df.columns or "name_hu" not in df.columns:
        df.columns = [c.lower() for c in df.columns]
    df["nace2"] = (
        df["nace2"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(2).str[:2]
    )
    df["name_hu"] = df["name_hu"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["nace2"].str.len() == 2].drop_duplicates("nace2")
    return dict(zip(df["nace2"], df["name_hu"]))

cs = load_cross_section()
labels = load_nace2_labels()

# ----------------------- UI: header ------------------------
st.title('Scatter — 2019 cross-section (simulated)')
st.markdown("Pick an **industry**, two **variables** and an optional **fit**. Monetary values are **1000 HUF**.")

# ----------------------- Sidebar ---------------------------
st.sidebar.header("Settings")

# Industry selector with names (+ ALL option)
codes = sorted(cs["nace2_2d"].unique(), key=lambda c: int(c))
opts = ["All industries (ALL)"]
code_from_label = {"All industries (ALL)": "ALL"}
for c in codes:
    name = labels.get(c, None)
    lab = f"{name} ({c})" if name else f"NACE {c} ({c})"
    opts.append(lab); code_from_label[lab] = c

def_idx = (1 + codes.index("10")) if "10" in codes else 0
sel_label = st.sidebar.selectbox("Industry (NACE2)", opts, index=def_idx)
code = code_from_label[sel_label]

# Variable menu — match simulated column names
var_map = {
    'Sales (1000 HUF)': 'sales_clean',
    'Tangible assets (1000 HUF)': 'tanass_clean',
    'Total assets (1000 HUF)': 'eszk',
    'Personal expenses (1000 HUF)': 'persexp_clean',
    'Pretax profit (1000 HUF)': 'pretax',
    'EBIT (1000 HUF)': 'ereduzem',
    'Export value (1000 HUF)': 'export_value',
    'Liabilities (1000 HUF)': 'liabilities',
    'Employment (headcount)': 'emp',
    'Age (years)': 'age',
}
available = {k: v for k, v in var_map.items() if v in cs.columns}
x_label = st.sidebar.selectbox("X variable", list(available.keys()), index=0)
y_label = st.sidebar.selectbox("Y variable", list(available.keys()), index=min(4, len(available)-1))
xvar = available[x_label]; yvar = available[y_label]

# Tails handling (X-axis)
st.sidebar.subheader("Tail handling (2% in X)")
winsor = st.sidebar.checkbox("Winsorize", value=True)
trim = st.sidebar.checkbox("Trim (exclude tails)", value=False)

# Fit type
fit_type = st.sidebar.selectbox(
    "Overlay fit",
    ["None", "Linear", "Quadratic", "Cubic", "LOWESS", "Stepwise (5 bins)", "Stepwise (20 bins)"],
    index=1
)
if fit_type == "LOWESS" and not HAS_LOWESS:
    st.sidebar.warning("statsmodels LOWESS not available; switch to another fit.")

alpha = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.5, 0.05)
size  = st.sidebar.slider("Point size", 5, 100, 20, 1)

# ----------------------- Filter & prep ---------------------
df = cs.copy() if code == "ALL" else cs[cs["nace2_2d"] == code].copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[xvar, yvar])

if df.empty:
    st.error("No data for the selected scope.")
    st.stop()

def apply_tails(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if len(s) < 5:
        return s
    q2, q98 = np.percentile(s, [2, 98])
    if trim:
        return s[(s > q2) & (s < q98)]
    return s.clip(q2, q98) if winsor else s

x = apply_tails(df[xvar])
y = df[yvar]
idx = x.index.intersection(y.index)
plot_df = pd.DataFrame({xvar: x.loc[idx], yvar: y.loc[idx]})

# ----------------------- Plot -----------------------------
fig, ax = plt.subplots()
sns.scatterplot(data=plot_df, x=xvar, y=yvar, s=size, alpha=alpha,
                edgecolor='white', linewidth=0.2, color=color[0], ax=ax)

# Fit overlays
if fit_type in {"Linear", "Quadratic", "Cubic"} and len(plot_df) >= 10:
    deg = {"Linear": 1, "Quadratic": 2, "Cubic": 3}[fit_type]
    xx = plot_df[xvar].to_numpy(); yy = plot_df[yvar].to_numpy()
    order = np.argsort(xx); xx_sorted = xx[order]; yy_sorted = yy[order]
    coef = np.polyfit(xx_sorted, yy_sorted, deg=deg)
    poly = np.poly1d(coef)
    xs = np.linspace(xx_sorted.min(), xx_sorted.max(), 400)
    ax.plot(xs, poly(xs), color=color[1], linewidth=2, alpha=0.9, label=f"{fit_type} fit")

elif fit_type == "LOWESS" and HAS_LOWESS and len(plot_df) >= 10:
    z = lowess(plot_df[yvar].values, plot_df[xvar].values, frac=0.25, return_sorted=True)
    ax.plot(z[:, 0], z[:, 1], color=color[1], linewidth=2, alpha=0.9, label="LOWESS fit")

elif fit_type.startswith("Stepwise") and len(plot_df) >= 10:
    steps = 5 if "5" in fit_type else 20
    plot_df["__bin"] = pd.qcut(plot_df[xvar], q=steps, duplicates="drop")
    gb = plot_df.groupby("__bin", observed=True)[[xvar, yvar]]
    bin_stats = gb.agg(x_min=(xvar, "min"), x_max=(xvar, "max"), y_mean=(yvar, "mean")).reset_index(drop=True)
    for _, r in bin_stats.iterrows():
        ax.hlines(y=r["y_mean"], xmin=r["x_min"], xmax=r["x_max"],
                  colors=color[1], linewidth=3, alpha=0.9)
    plot_df.drop(columns="__bin", inplace=True)

# Axes labels & formatting
ax.set_xlabel(x_label); ax.set_ylabel(y_label)
ax.spines[['top', 'right']].set_visible(False)
ax.ticklabel_format(style='plain', axis='x')
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
ax.tick_params(axis='x', labelrotation=25)

plt.tight_layout()
st.pyplot(fig)

# ----------------------- Summary --------------------------
scope_label = "All industries" if code == "ALL" else f"{labels.get(code, f'NACE {code}')} ({code})"
tail_note = "trim" if trim else ("winsorize" if winsor else "raw")
st.markdown(
    f"**Scope:** {scope_label} · **X:** `{x_label}` · **Y:** `{y_label}` · "
    f"**Obs:** {len(plot_df):,} · **Tails:** {tail_note} · **Fit:** {fit_type}"
)
