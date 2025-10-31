import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter

color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]
st.set_page_config(page_title='Visualizing Distributions — Firms (HU, simulated)', layout='wide')

@st.cache_data
def load_cross_section(path: str = 'data/synthetic/sim_cs2019_by_nace2_withcats.parquet') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {p}")
        st.stop()
    df = pd.read_parquet(p).copy()
    if "nace2_name_code" not in df.columns:
        st.error("Column `nace2_name_code` is missing from the data.")
        st.stop()
    return df

cs = load_cross_section()

# ----------------------------- Header ------------------------------
st.title('Visualizing Distributions — 2019 cross-section (simulated)')
st.markdown("Choose an **industry** and a **variable** to visualize. Monetary values shown in **million HUF**.")

# ----------------------------- Sidebar -----------------------------
st.sidebar.header('Settings')

# Build industry options: ALL first, then sorted by numeric code in the label
lab_df = pd.DataFrame({"label": cs["nace2_name_code"].dropna().unique()})
lab_df["__code"] = pd.to_numeric(lab_df["label"].str.extract(r"\((\d{1,2})\)\s*$", expand=False), errors="coerce")
lab_df = lab_df.sort_values(["__code", "label"]).drop(columns="__code")

labels = ["All industries (ALL)"] + lab_df["label"].tolist()
def_idx = 1 if len(labels) > 1 else 0
selected_label = st.sidebar.selectbox("Industry", labels, index=def_idx)

# Variable selection — monetary variables will be displayed in million HUF
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

available = [k for k, v in var_map.items() if v in cs.columns]
if not available:
    st.error("None of the expected variables were found in the data.")
    st.stop()

var_label = st.sidebar.selectbox('Variable to plot', available, index=0)
var = var_map[var_label]
is_monetary = var in MONETARY_VARS.values()

# Tail handling (2% each side)
st.sidebar.subheader("Tail handling (2% each side)")
winsorize = st.sidebar.checkbox("Winsorize top & bottom 2%", value=True)
trim = st.sidebar.checkbox("Exclude top & bottom 2%", value=False)

# Bins only
st.sidebar.subheader('Histogram Settings')
bins = st.sidebar.slider('Number of bins', min_value=5, max_value=60, value=25, step=1)

# ----------------------------- Filter ------------------------------
if selected_label == "All industries (ALL)":
    workset = cs.copy()
else:
    workset = cs[cs["nace2_name_code"] == selected_label].copy()

if workset.empty:
    st.error('No data for the selected filters.')
    st.stop()

# scale to million HUF when monetary
x_raw = workset[var].replace([np.inf, -np.inf], np.nan).dropna()
x = (x_raw / 1000.0) if is_monetary else x_raw

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

# No scientific notation + tilted labels + separators
ax.ticklabel_format(style='plain', axis='x')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.2f}" if is_monetary else f"{x:,.0f}"))
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
        try:
            return f"{v:,.2f}" if is_monetary else f"{v:,.0f}"
        except Exception:
            return str(v)
    stats_df['Value'] = stats_df['Value'].map(_fmt)
    st.dataframe(stats_df, use_container_width=True)
else:
    st.info('No bars with ≥5 observations remained to summarize.')

# ----------------------------- Summary ------------------------------
scope_label = selected_label
tail_note = "trimmed (excluded)" if trim else ("winsorized" if winsorize else "raw")
unit_note = "million HUF" if is_monetary else "raw units"
st.markdown(
    f"**Scope:** {scope_label} · **Variable:** `{var_label}` · "
    f"**Obs (plotted):** {len(x_plot):,} · **Tails:** {tail_note} · **Bins:** {bins} · "
    f"**Units:** {unit_note} · **Bars with <5 obs are hidden.**"
)
