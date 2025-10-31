import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import FuncFormatter

color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]
st.set_page_config(page_title='Visualizing Distributions — Firms (HU, simulated)', layout='wide')

# ----------------------------- Loaders -----------------------------
@st.cache_data
def load_cross_section(path: str = 'data/synthetic/sim_cs2019_by_nace2_withcats.parquet') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {p}")
        st.stop()
    df = pd.read_parquet(p).copy()
    # normalize 2-digit NACE code
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
nace_labels = load_nace2_labels()

# ----------------------------- Header ------------------------------
st.title('Visualizing Distributions — 2019 cross-section (simulated)')
st.markdown("Choose an **industry (NACE2)** and a **variable** to visualize. All monetary values are **1000 HUF**.")

# ----------------------------- Sidebar -----------------------------
st.sidebar.header('Settings')

# Industry dropdown with “ALL” and Name (code)
codes = sorted(cs["nace2_2d"].unique(), key=lambda c: int(c))
options = ["All industries (ALL)"]
code_for_label = {"All industries (ALL)": "ALL"}
for c in codes:
    name = nace_labels.get(c, None)
    label = f"{name} ({c})" if name else f"NACE {c} ({c})"
    options.append(label); code_for_label[label] = c
def_ind = (1 + codes.index("10")) if "10" in codes else 0
selected_label = st.sidebar.selectbox("Industry (NACE2)", options, index=def_ind)
industry_code = code_for_label[selected_label]

# Variable selection — use simulated column names
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
available = [k for k, v in var_map.items() if v in cs.columns]
if not available:
    st.error("None of the expected variables were found in the data.")
    st.stop()
var_label = st.sidebar.selectbox('Variable to plot', available, index=0)
var = var_map[var_label]

# Tail handling (2% each side)
st.sidebar.subheader("Tail handling (2% each side)")
winsorize = st.sidebar.checkbox("Winsorize top & bottom 2%", value=True)
trim = st.sidebar.checkbox("Exclude top & bottom 2%", value=False)

# Bins only
st.sidebar.subheader('Histogram Settings')
bins = st.sidebar.slider('Number of bins', min_value=5, max_value=60, value=25, step=1)

# ----------------------------- Filter ------------------------------
workset = cs.copy() if industry_code == "ALL" else cs[cs['nace2_2d'] == industry_code].copy()
if workset.empty:
    st.error('No data for the selected filters.')
    st.stop()

x = workset[var].replace([np.inf, -np.inf], np.nan).dropna()
if x.empty:
    st.warning("Selected variable has no valid data after filtering.")
    st.stop()

# Tails on the filtered set
if len(x) >= 5:
    q2, q98 = np.percentile(x, [2, 98])
else:
    q2, q98 = np.nanmin(x), np.nanmax(x)

if trim:
    x_plot = x[(x >= q2) & (x <= q98)]
else:
    x_plot = x.clip(q2, q98) if winsorize else x

# ----------------------------- Plot -------------------------------
fig, ax = plt.subplots()

# Compute histogram and hide small bins (<5 obs)
counts, edges = np.histogram(x_plot, bins=bins)
counts_masked = counts.copy()
counts_masked[counts_masked < 5] = 0

widths = np.diff(edges)
ax.bar(edges[:-1], counts_masked, width=widths, align='edge',
       color=color[0], edgecolor='white', linewidth=0.5)

st.subheader(f'Histogram of {var_label} — {bins} bins (bars with <5 obs are hidden)')
ax.set_xlabel(var_label)
ax.set_ylabel('Frequency')
ax.spines[['top', 'right']].set_visible(False)

# No scientific notation + tilted labels
ax.ticklabel_format(style='plain', axis='x')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
ax.tick_params(axis='x', labelrotation=25)

plt.tight_layout()
st.pyplot(fig)

# ----------------------- Descriptive statistics ----------------------
x_stats = x_plot.replace([np.inf, -np.inf], np.nan).dropna()
if not x_stats.empty:
    modes = x_stats.mode(dropna=True)
    mode_val = modes.iloc[0] if len(modes) > 0 else np.nan
    stats = {
        'min': float(np.nanmin(x_stats)),
        'max': float(np.nanmax(x_stats)),
        'range': float(np.nanmax(x_stats) - np.nanmin(x_stats)),
        'mean': float(np.nanmean(x_stats)),
        'median': float(np.nanmedian(x_stats)),
        'mode': float(mode_val) if pd.api.types.is_numeric_dtype(x_stats) else mode_val,
        'std (sample)': float(x_stats.std(ddof=1)),
        'variance (sample)': float(x_stats.var(ddof=1)),
        'skewness': float(x_stats.skew())
    }
    st.subheader('Descriptive statistics (of plotted data)')
    stats_df = pd.DataFrame({'Statistic': list(stats.keys()), 'Value': list(stats.values())})
    def _fmt(v):
        try: return f"{v:,.2f}"
        except Exception: return str(v)
    stats_df['Value'] = stats_df['Value'].map(_fmt)
    st.dataframe(stats_df, use_container_width=True)
else:
    st.info('No bars with ≥5 observations remained to summarize.')

# ----------------------------- Summary ------------------------------
scope_label = "All industries" if industry_code == "ALL" else f"{nace_labels.get(industry_code, f'NACE {industry_code}')} ({industry_code})"
tail_note = "trimmed (excluded)" if trim else ("winsorized" if winsorize else "raw")
st.markdown(
    f"**Scope:** {scope_label} · **Variable:** `{var_label}` · "
    f"**Obs (plotted):** {len(x_plot):,} · **Tails:** {tail_note} · **Bins:** {bins} · "
    f"**Bars with <5 obs are hidden.**"
)
