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

color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]
st.set_page_config(page_title='Scatter — Firms (HU cross-section, simulated)', layout='wide')

@st.cache_data
def load_cross_section(path: str = 'data/synthetic/sim_cs2019_by_nace2_withcats.parquet') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {p}")
        st.stop()
    df = pd.read_parquet(p).copy()
    need = {"nace2", "nace2_name_code"}
    missing = need - set(df.columns)
    if missing:
        st.error(f"Missing columns in data: {missing}")
        st.stop()
    df["nace2"] = df["nace2"].astype(str)
    df["nace2_name_code"] = df["nace2_name_code"].astype(str)
    return df

cs = load_cross_section()

# ----------------------- UI: header ------------------------
st.title('Scatter — 2019 cross-section (simulated)')
st.markdown("Pick an **industry**, two **variables** and an optional **fit**. Monetary values shown in **million HUF**.")

# ----------------------- Sidebar ---------------------------
st.sidebar.header("Settings")

# Industry options: ALL first, then by code
lab_df = pd.DataFrame({"label": cs["nace2_name_code"].dropna().unique()})
lab_df["__code"] = pd.to_numeric(lab_df["label"].str.extract(r"\((\d{1,2})\)\s*$", expand=False), errors="coerce")
lab_df = lab_df.sort_values(["__code", "label"]).drop(columns="__code")
opts = ["All industries (ALL)"] + lab_df["label"].tolist()

sel_label = st.sidebar.selectbox("Industry", opts, index=0)
scope_all = sel_label == "All industries (ALL)"

# Variables (mark monetary with “(million HUF)”)
MONETARY_VARS = {
    'Sales (million HUF)': 'sales_clean',
    'Tangible assets (million HUF)': 'tanass_clean',
    'Total assets (million HUF)': 'eszk',
    'Personal expenses (million HUF)': 'persexp_clean',
    'Pretax profit (million HUF)': 'pretax',
    'EBIT (million HUF)': 'ereduzem',
    'Export value (million HUF)': 'export_value',
    'Liabilities (million HUF)': 'liabilities',
}
NON_MONETARY_VARS = {
    'Employment (headcount)': 'emp',
    'Age (years)': 'age',
}
var_map = {**MONETARY_VARS, **NON_MONETARY_VARS}

available = {k: v for k, v in var_map.items() if v in cs.columns}
x_label = st.sidebar.selectbox("X variable", list(available.keys()), index=0)
y_label = st.sidebar.selectbox("Y variable", list(available.keys()), index=min(4, len(available)-1))
xvar = available[x_label]; yvar = available[y_label]
x_is_monetary = xvar in MONETARY_VARS.values()
y_is_monetary = yvar in MONETARY_VARS.values()

# Tails handling (2% tails) — X and Y separately
st.sidebar.subheader("Tail handling (2% tails)")
winsor_x = st.sidebar.checkbox("Winsorize X", value=True)
trim_x   = st.sidebar.checkbox("Trim X (exclude tails)", value=False)
winsor_y = st.sidebar.checkbox("Winsorize Y", value=True)
trim_y   = st.sidebar.checkbox("Trim Y (exclude tails)", value=False)

# Fit type
fit_type = st.sidebar.selectbox(
    "Overlay fit",
    ["None", "Linear", "Quadratic", "Cubic", "LOWESS", "Stepwise (5 bins)", "Stepwise (20 bins)"],
    index=1
)
if fit_type == "LOWESS" and not HAS_LOWESS:
    st.sidebar.warning("statsmodels LOWESS not available; switch to another fit.")

# Plot cosmetics
alpha = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.5, 0.05)
size  = st.sidebar.slider("Point size", 5, 100, 20, 1)

# Log scales
logx = st.sidebar.checkbox("Log scale X", value=False)
logy = st.sidebar.checkbox("Log scale Y", value=False)

# ----------------------- Filter & prep ---------------------
df = cs.copy() if scope_all else cs[cs["nace2_name_code"] == sel_label].copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[xvar, yvar])

# scale to million HUF for monetary variables (before tail handling)
if x_is_monetary:
    df[xvar] = df[xvar] / 1000.0
if y_is_monetary:
    df[yvar] = df[yvar] / 1000.0

# If a log axis is requested, drop non-positive values on that axis (after scaling)
if logx:
    df = df[df[xvar] > 0]
if logy:
    df = df[df[yvar] > 0]

if df.empty:
    st.error("No data for the selected scope.")
    st.stop()

def apply_tails(s: pd.Series, do_winsor: bool, do_trim: bool) -> pd.Series:
    s = s.dropna()
    if len(s) < 5:
        return s
    q2, q98 = np.percentile(s, [2, 98])
    if do_trim:
        return s[(s > q2) & (s < q98)]
    return s.clip(q2, q98) if do_winsor else s

x = apply_tails(df[xvar], winsor_x, trim_x)
y = apply_tails(df[yvar], winsor_y, trim_y)

# align indices (keep rows present in both after tail handling)
idx = x.index.intersection(y.index)
plot_df = pd.DataFrame({xvar: x.loc[idx], yvar: y.loc[idx]})

if plot_df.empty:
    st.error("No data after tail handling. Try relaxing trim/winsor or log settings.")
    st.stop()

# ----------------------- Plot -----------------------------
fig, ax = plt.subplots()
sns.scatterplot(data=plot_df, x=xvar, y=yvar, s=size, alpha=alpha,
                edgecolor='white', linewidth=0.2, color=color[0], ax=ax)

coef_text = None  # will populate for Linear/Quadratic/Cubic

# Fit overlays (on displayed scale)
if fit_type in {"Linear", "Quadratic", "Cubic"} and len(plot_df) >= 10:
    deg = {"Linear": 1, "Quadratic": 2, "Cubic": 3}[fit_type]
    xx = plot_df[xvar].to_numpy(); yy = plot_df[yvar].to_numpy()
    order = np.argsort(xx); xx_sorted = xx[order]; yy_sorted = yy[order]
    coef = np.polyfit(xx_sorted, yy_sorted, deg=deg)
    poly = np.poly1d(coef)
    xs = np.linspace(xx_sorted.min(), xx_sorted.max(), 400)
    ax.plot(xs, poly(xs), color=color[1], linewidth=2, alpha=0.9, label=f"{fit_type} fit")

    if deg == 1:
        a, b = coef[1], coef[0]
        coef_text = f"Linear fit: y = {a:.4g} + {b:.4g}·x"
    elif deg == 2:
        a, b, c = coef[2], coef[1], coef[0]
        coef_text = f"Quadratic fit: y = {a:.4g} + {b:.4g}·x + {c:.4g}·x²"
    else:
        a, b, c, d = coef[3], coef[2], coef[1], coef[0]
        coef_text = f"Cubic fit: y = {a:.4g} + {b:.4g}·x + {c:.4g}·x² + {d:.4g}·x³"

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

# ticks: no scientific notation + separators; decimals for monetary, integers otherwise
ax.ticklabel_format(style='plain', axis='x')
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}" if x_is_monetary else f"{v:,.0f}"))
ax.tick_params(axis='x', labelrotation=25)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}" if y_is_monetary else f"{v:,.0f}"))

# log toggles
if logx:
    ax.set_xscale('log')
if logy:
    ax.set_yscale('log')

plt.tight_layout()
st.pyplot(fig)

# ----------------------- Summary --------------------------
scope_label = sel_label
tail_note_x = "trim" if trim_x else ("winsorize" if winsor_x else "raw")
tail_note_y = "trim" if trim_y else ("winsorize" if winsor_y else "raw")

st.markdown(
    f"**Scope:** {scope_label} · **X:** `{x_label}` · **Y:** `{y_label}` · "
    f"**Obs:** {len(plot_df):,} · **Tails (X/Y):** {tail_note_x} / {tail_note_y} · "
    f"**Fit:** {fit_type} · **Log (X/Y):** {logx} / {logy}"
)

if coef_text is not None:
    st.markdown(f"**Fit parameters:** {coef_text}")
