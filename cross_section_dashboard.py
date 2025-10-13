import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------
# Look & feel
# ------------------------------------------------------
color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]
st.set_page_config(page_title='Visualizing Distributions — Firms (HU)', layout='wide')

# ------------------------------------------------------
# Data loading
# ------------------------------------------------------
@st.cache_data
def load_cross_section(path: str = 'data/synthetic/synthetic_cs2019_from_seq_regs_stdnames.parquet') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {p}")
        st.stop()
    df = pd.read_parquet(p)

    # enforce dtypes / friendly
    df = df.copy()
    if "nace2" in df.columns:
        # show as 2-digit string
        df["nace2"] = df["nace2"].astype(float).astype(int).astype(str)
    if "firm_owner" in df.columns:
        df["firm_owner"] = df["firm_owner"].astype("category")
    if "has_subsidy" in df.columns:
        df["has_subsidy"] = df["has_subsidy"].astype(int)
    if "emp" in df.columns:
        df["emp"] = df["emp"].astype(int)
    return df

cs = load_cross_section()

# ------------------------------------------------------
# Title & description
# ------------------------------------------------------
st.title('Visualizing Distributions — Firms (2019 cross-section)')
st.markdown(
    """
Use the sidebar to choose an **industry (NACE2)**, an **employment range**, and a **variable** to visualize.
All monetary variables are in **1000 HUF**.
""")

# ------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------
st.sidebar.header('Settings')

# Industry (NACE2)
if "nace2" not in cs.columns:
    st.error("Column `nace2` not found in the data.")
    st.stop()

industries = sorted(cs["nace2"].unique())
def_ind = industries.index("10") if "10" in industries else 0
industry = st.sidebar.selectbox('Industry (NACE2)', industries, index=def_ind)

# Employment range (emp)
if "emp" not in cs.columns:
    st.error("Column `emp` not found in the data.")
    st.stop()

emp_min = int(cs["emp"].min())
emp_max = int(cs["emp"].max())
emp_range = st.sidebar.slider('Employment range', min_value=emp_min, max_value=emp_max,
                              value=(max(emp_min, 1), min(emp_max, 250)), step=1)

# Variable selection
var_map = {
    'Sales (1000 HUF)': 'sales_1000HUF',
    'Tangible assets (1000 HUF)': 'tangible_assets_1000HUF',
    'Personal expenses (1000 HUF)': 'personal_expenses_1000HUF',
    'EBIT (1000 HUF)': 'EBIT_1000HUF',
    'Export value (1000 HUF)': 'export_value_1000HUF',
    'Profit (1000 HUF)': 'profit_1000HUF',
    'Grant received (1000 HUF)': 'grant_1000HUF',
    'Equity ratio': 'eq_ratio',
    'Employment (headcount)': 'emp',
}
available = [k for k, v in var_map.items() if v in cs.columns]
if not available:
    st.error("None of the expected variables were found in the data.")
    st.stop()

var_label = st.sidebar.selectbox('Variable to plot', available, index=0)
var = var_map[var_label]

# --- Value filter for the selected variable (numeric slider) ---
# Compute data-driven bounds from the full cross-section to avoid empty sliders
_vals_all = cs[var].replace([np.inf, -np.inf], np.nan).dropna()
if _vals_all.empty:
    st.sidebar.info(f"No valid values found for `{var_label}`; skipping value filter.")
    value_range = None
else:
    vmin = float(_vals_all.min())
    vmax = float(_vals_all.max())
    # use p1/p99 to make the default range sensible but allow full range
    p1, p99 = np.percentile(_vals_all, [1, 99])
    value_range = st.sidebar.slider(
        f"Filter {var_label} (min–max)",
        min_value=float(vmin),
        max_value=float(vmax),
        value=(float(p1), float(p99)),
        step=float((vmax - vmin) / 500 or 1.0)
    )

# Histogram settings
st.sidebar.subheader('Histogram Settings')
bins = st.sidebar.slider('Number of bins', min_value=5, max_value=500, value=50, step=1)
bin_width = st.sidebar.slider('Bin width', min_value=1, max_value=100000, value=1000, step=1)
binning_option = st.sidebar.selectbox('Binning option', ['Number of bins', 'Bin width'], index=0)

# ------------------------------------------------------
# Filter data
# ------------------------------------------------------
workset = cs[
    (cs['nace2'] == str(industry)) &
    (cs['emp'] >= emp_range[0]) &
    (cs['emp'] <= emp_range[1])
].copy()

if value_range is not None:
    workset = workset[(workset[var] >= value_range[0]) & (workset[var] <= value_range[1])]

if workset.empty:
    st.error('No data for the selected filters.')
    st.stop()

# ------------------------------------------------------
# Main layout — Histogram + descriptive stats
# ------------------------------------------------------
col1, _ = st.columns([2, 1])
with col1:
    x = workset[var].replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        st.warning("Selected variable has no valid data after filtering.")
    else:
        fig, ax = plt.subplots()

        # Helper: enforce "≥5 obs per bin"
        adjusted = ""
        if binning_option == 'Number of bins':
            current_bins = bins
            while current_bins > 1:
                counts, edges = np.histogram(x, bins=current_bins)
                if (counts >= 5).all():
                    break
                current_bins = max(1, int(np.floor(current_bins * 0.8)))
            if current_bins != bins:
                adjusted = f" (adjusted to {current_bins} to keep ≥5 obs/bin)"
            sns.histplot(x, bins=max(current_bins, 1), kde=False, ax=ax,
                         color=color[0], edgecolor='white', fill=True, alpha=1)
            st.subheader(f'Histogram of {var_label} — {max(current_bins,1)} bins{adjusted}')
        else:
            current_bw = bin_width
            # try increasing bin width until all bins have ≥5 obs (cap at range)
            data_range = (float(x.min()), float(x.max()))
            max_bw = max(1.0, (data_range[1] - data_range[0]) / 5.0)
            tries = 0
            while tries < 20:
                counts, edges = np.histogram(x, bins=np.arange(data_range[0], data_range[1] + current_bw, current_bw))
                if len(counts) == 0 or (counts >= 5).all() or current_bw >= max_bw:
                    break
                current_bw *= 1.3
                tries += 1
            if current_bw != bin_width:
                adjusted = f" (adjusted to ~{int(current_bw)} to keep ≥5 obs/bin)"
            sns.histplot(x, binwidth=current_bw, kde=False, ax=ax,
                         color=color[0], edgecolor='white', fill=True, alpha=1)
            st.subheader(f'Histogram of {var_label} — bin width ≈ {int(current_bw)}{adjusted}')

        ax.set_xlabel(var_label)
        ax.set_ylabel('Frequency')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

# ------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------
x = workset[var].replace([np.inf, -np.inf], np.nan).dropna()
if not x.empty:
    modes = x.mode(dropna=True)
    mode_val = modes.iloc[0] if len(modes) > 0 else np.nan

    stats = {
        'min': float(np.nanmin(x)),
        'max': float(np.nanmax(x)),
        'range': float(np.nanmax(x) - np.nanmin(x)),
        'mean': float(np.nanmean(x)),
        'median': float(np.nanmedian(x)),
        'mode': float(mode_val) if pd.api.types.is_numeric_dtype(x) else mode_val,
        'std (sample)': float(x.std(ddof=1)),
        'variance (sample)': float(x.var(ddof=1)),
        'skewness': float(x.skew())
    }
    st.subheader('Descriptive statistics')
    stats_df = pd.DataFrame({'Statistic': list(stats.keys()), 'Value': list(stats.values())})

    def _fmt(v):
        try:
            return f"{v:,.2f}"
        except Exception:
            return str(v)

    stats_df['Value'] = stats_df['Value'].map(_fmt)
    st.dataframe(stats_df, use_container_width=True)
else:
    st.info('Not enough data to compute descriptive statistics for the current selection.')

# ------------------------------------------------------
# Tiny summary
# ------------------------------------------------------
st.markdown(
    f"**Filters:** NACE2 `{industry}`, emp in [{emp_range[0]}, {emp_range[1]}]. "
    f"**Obs:** {len(workset):,}."
)
