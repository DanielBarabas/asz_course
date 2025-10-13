import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------
# Look & feel
# ------------------------------------------------------
color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0", "#8c3a3a"]
st.set_page_config(page_title='Comparing Years — Firms Panel (Food, HU)', layout='wide')

# ------------------------------------------------------
# Data loading
# ------------------------------------------------------
@st.cache_data
def load_panel(path: str = 'data/synthetic/panel_food_simulated.parquet') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {p}")
        st.stop()
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df.copy()
    # enforce expected dtypes & naming
    if "year" in df.columns:   df["year"] = df["year"].astype(int)
    if "nace2" in df.columns:  df["nace2"] = df["nace2"].astype(str)
    if "firm_owner" in df.columns: df["firm_owner"] = df["firm_owner"].astype("category")
    if "has_export" in df.columns: df["has_export"] = df["has_export"].astype(int)
    if "emp" in df.columns:        df["emp"] = df["emp"].astype(int)
    return df

panel = load_panel()

# Limit to Food industry (NACE2 == "10") if column present
if "nace2" in panel.columns:
    panel = panel[panel["nace2"] == "10"]

# ------------------------------------------------------
# Title & description
# ------------------------------------------------------
st.title('Two-Year Scatter Comparison — Food industry panel')
st.markdown("Compare **two years** for **two variables** with separate colors and OLS lines. "
            "All monetary values are in **1000 HUF**.")

# ------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------
st.sidebar.header('Settings')

years = sorted(panel['year'].unique())
idx1 = years.index(2010) if 2010 in years else 0
idx2 = years.index(2022) if 2022 in years else len(years) - 1

year1 = st.sidebar.selectbox('Year A', years, index=idx1)
year2 = st.sidebar.selectbox('Year B', years, index=idx2)

# Your actual column names
all_vars = {
    'Sales (1000 HUF)': 'sales_clean',
    'Total assets (1000 HUF)': 'eszk',
    'Tangible assets (1000 HUF)': 'tanass_clean',
    'Personal expenses (1000 HUF)': 'persexp_clean',
    'Profit (1000 HUF)': 'pretax',
    'EBIT (1000 HUF)': 'ereduzem',
    'Export value (1000 HUF)': 'export',
    'Grant received (1000 HUF)': 'grant_1000HUF',  # present only if you added it
    'Employment (headcount)': 'emp',
}
# Keep only variables that exist in the data
var_map = {k: v for k, v in all_vars.items() if v in panel.columns}

x_label = st.sidebar.selectbox('X variable', list(var_map.keys()), index=0)
y_label = st.sidebar.selectbox('Y variable', list(var_map.keys()),
                               index=min(4, len(var_map) - 1))

xvar = var_map[x_label]
yvar = var_map[y_label]

st.sidebar.subheader('Plot Options')
alpha = st.sidebar.slider('Point opacity', 0.1, 1.0, 0.5, 0.05)
size  = st.sidebar.slider('Point size', 5, 100, 20, 1)
logx  = st.sidebar.checkbox('Log scale X', value=False)
logy  = st.sidebar.checkbox('Log scale Y', value=False)
max_points = st.sidebar.number_input('Max points per year', 500, 50000, 20000, 500)

# ------------------------------------------------------
# Filter & value-range sliders
# ------------------------------------------------------
sel = panel[panel['year'].isin([year1, year2])].copy()

# Value-range sliders (use 1–99 percentiles as defaults)
def slider_for(col, label):
    s = sel[col].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    vmin, vmax = float(s.min()), float(s.max())
    p1, p99 = np.percentile(s, [1, 99])
    return st.sidebar.slider(
        f'Filter {label} (min–max)',
        min_value=float(vmin),
        max_value=float(vmax),
        value=(float(p1), float(p99)),
        step=float((vmax - vmin) / 500 or 1.0)
    )

st.sidebar.subheader('Value filters')
xrng = slider_for(xvar, x_label)
yrng = slider_for(yvar, y_label)

# Drop non-finite; if log requested, require positive values
sel = sel.replace([np.inf, -np.inf], np.nan)
if logx: sel = sel[sel[xvar] > 0]
if logy: sel = sel[sel[yvar] > 0]
sel = sel.dropna(subset=[xvar, yvar])

# Apply value filters
if xrng is not None:
    sel = sel[(sel[xvar] >= xrng[0]) & (sel[xvar] <= xrng[1])]
if yrng is not None:
    sel = sel[(sel[yvar] >= yrng[0]) & (sel[yvar] <= yrng[1])]

# Downsample per year if needed
parts = []
for y in (year1, year2):
    sub = sel[sel['year'] == y]
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)
    parts.append(sub)
workset = pd.concat(parts, ignore_index=True)

st.sidebar.download_button(
    label='Download selection as CSV',
    data=workset.to_csv(index=False).encode('utf-8'),
    file_name=f'panel_food_{xvar}_vs_{yvar}_{year1}_{year2}.csv',
    mime='text/csv'
)

if workset.empty:
    st.error('No data available for the selected filters.')
    st.stop()

# ------------------------------------------------------
# Plot
# ------------------------------------------------------
fig, ax = plt.subplots()
palette = {year1: color[0], year2: color[1]}

for y in [year1, year2]:
    sub = workset[workset['year'] == y]
    if sub.empty: 
        continue
    sns.scatterplot(data=sub, x=xvar, y=yvar, ax=ax,
                    s=size, alpha=alpha, edgecolor='white', linewidth=0.2,
                    color=palette[y], label=str(y))
    sns.regplot(data=sub, x=xvar, y=yvar, ax=ax,
                scatter=False, color=palette[y],
                line_kws={'linewidth': 2, 'alpha': 0.9})

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.spines[['top', 'right']].set_visible(False)
if logx: ax.set_xscale('log')
if logy: ax.set_yscale('log')
ax.legend(title='Year')

plt.tight_layout()
st.pyplot(fig)

# ------------------------------------------------------
# Summary
# ------------------------------------------------------
st.markdown(
    f"**Years:** {year1} vs {year2}. **Variables:** X=`{x_label}`, Y=`{y_label}`. "
    f"**Obs:** {len(workset):,} (cap {int(max_points):,}/year)."
)
