import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

# ------------------------------------------------------
# Config
# ------------------------------------------------------
st.set_page_config(page_title="Growth Regressions — Firms (simulated)", layout="wide")
st.title("Growth Regressions — 2019 cross-section (simulated)")

@st.cache_data
def load_cs(path: str = "data/synthetic/sim_cs2019_by_nace2_withcats.parquet") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {p}")
        st.stop()
    df = pd.read_parquet(p).copy()

    # sanity: must have industry label used for filtering
    if "nace2_name_code" not in df.columns:
        st.error("Column `nace2_name_code` is missing from the data.")
        st.stop()

    # coerce types
    if "nace2" in df.columns:
        df["nace2"] = df["nace2"].astype(str)

    # we’ll compute ln_sales if we need it
    if "sales_clean" in df.columns:
        df["ln_sales"] = np.log(np.clip(df["sales_clean"].astype(float), 1e-9, None))

    return df

df = load_cs()

st.markdown(
    """
Choose an **industry**, an **outcome**, and **regressors**.  
Outcomes supported:

- **Relative growth**: (sales_lead − sales) / sales  
- **Log growth**: ln(sales_lead) − ln(sales)

"""
)

# ------------------------------------------------------
# Sidebar: industry filter
# ------------------------------------------------------
st.sidebar.header("Filters & Model")

# Sort industries by numeric code inside the label, “ALL” first
lab_df = pd.DataFrame({"label": df["nace2_name_code"].dropna().unique()})
lab_df["__code"] = pd.to_numeric(lab_df["label"].str.extract(r"\((\d{1,2})\)\s*$", expand=False),
                                 errors="coerce")
lab_df = lab_df.sort_values(["__code", "label"]).drop(columns="__code")
industry_opts = ["All industries (ALL)"] + lab_df["label"].tolist()

sel_industry = st.sidebar.selectbox("Industry", industry_opts, index=0)
if sel_industry == "All industries (ALL)":
    d = df.copy()
else:
    d = df[df["nace2_name_code"] == sel_industry].copy()


# --- Employment size classes from `emp`
if "emp" in d.columns:
    emp_float = pd.to_numeric(d["emp"], errors="coerce")
    d["emp_size"] = pd.cut(
        emp_float,
        bins=[-np.inf, 5, 50, 250, np.inf],
        labels=["≤5", "5–50", "50–250", "250+"],
        right=True,
        ordered=True
    ).astype("category")
else:
    d["emp_size"] = pd.Series(pd.Categorical([np.nan]*len(d)))


# ------------------------------------------------------
# Outcomes
# ------------------------------------------------------
outcome_choice = st.sidebar.selectbox(
    "Outcome",
    ["Relative growth: (sales_lead - sales)/sales", "Log growth: ln(sales_lead) - ln(sales)"],
    index=1
)

# Determine “lead” values
sales = d["sales_clean"] if "sales_clean" in d.columns else None
sales_lead = None
ln_sales = d["ln_sales"] if "ln_sales" in d.columns else None
ln_sales_lead = None

if "sales_lead_sim" in d.columns:
    sales_lead = d["sales_lead_sim"].astype(float)
if sales_lead is None and "sales22_lead2" in d.columns:
    sales_lead = d["sales22_lead2"].astype(float)

if "ln_sales_lead_sim" in d.columns:
    ln_sales_lead = d["ln_sales_lead_sim"].astype(float)
if ln_sales_lead is None and "ln_sales22_lead2" in d.columns:
    ln_sales_lead = d["ln_sales22_lead2"].astype(float)

# Compute outcomes
y = None
if outcome_choice.startswith("Relative"):
    if sales is None or sales_lead is None:
        st.error("Cannot compute relative growth: need `sales_clean` and a lead (`sales_lead_sim` or `sales22_lead2`).")
        st.stop()
    base = sales.astype(float)
    y = (sales_lead - base) / np.where(base != 0, base, np.nan)

else:  # log growth
    # derive ln_sales if missing and sales available
    if ln_sales is None and sales is not None:
        ln_sales = np.log(np.clip(sales.astype(float), 1e-9, None))

    if ln_sales is None or ln_sales_lead is None:
        st.error("Cannot compute log growth: need `ln_sales` (or `sales_clean`) and a log lead (`ln_sales_lead_sim` or `ln_sales22_lead2`).")
        st.stop()
    y = ln_sales_lead - ln_sales

# Clean outcome
y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------
# RHS pickers  (auto small-cardinality categoricals + per-var quad/log + interactions)
# ------------------------------------------------------
exclude_cols = {
    "nace2_name_code", "nace2", "growth_sim", "ln_sales_lead_sim", "sales_lead_sim",
    "ln_sales22_lead2", "sales22_lead2", "ln_sales", "name_hu", "row_id","exit","county"
}
MAX_CAT_LEVELS = 10

is_num  = pd.api.types.is_numeric_dtype
is_bool = pd.api.types.is_bool_dtype

# Classify columns:
candidate_cols = [c for c in d.columns if c not in exclude_cols]
categorical_cols, numeric_cols = [], []
for c in candidate_cols:
    s = d[c]
    nun = s.nunique(dropna=True)
    if (
        is_bool(s) or
        s.dtype == "object" or
        pd.api.types.is_categorical_dtype(s) or
        nun <= MAX_CAT_LEVELS
    ):
        categorical_cols.append(c)
    elif is_num(s):
        numeric_cols.append(c)
# Note: emp_size naturally falls into categorical_cols (4 levels)

st.sidebar.subheader("Regressors (RHS)")
cont_vars = st.sidebar.multiselect("Continuous regressors", options=sorted(numeric_cols))
cat_vars  = st.sidebar.multiselect("Categorical regressors", options=sorted(categorical_cols))

# --- Per-variable quadratic toggles
st.sidebar.markdown("**Quadratic terms (choose per continuous variable):**")
quad_selected = []
for v in cont_vars:
    if st.sidebar.checkbox(f"Include {v}²", value=False, key=f"quad__{v}"):
        quad_selected.append(v)
quad_set = set(quad_selected)

# --- Per-variable log toggles
st.sidebar.markdown("**Log transforms (choose per continuous variable):**")
log_selected = []
for v in cont_vars:
    if st.sidebar.checkbox(f"Include log({v})", value=False, key=f"log__{v}"):
        log_selected.append(v)
log_set = set(log_selected)

st.sidebar.markdown("**Interaction:**")
# --- Optional interaction with ownership (emp_size × ownership)
ownership_default_idx = 0
ownership_options = [c for c in sorted(set(["firm_owner"]) | set(categorical_cols)) if c in d.columns]
if "firm_owner" in ownership_options:
    ownership_default_idx = ownership_options.index("firm_owner")

interact_emp_owner = st.sidebar.checkbox("Interact employment size (emp_size) with ownership", value=False)
owner_var = "firm_owner"

rhs_cols = cont_vars + cat_vars
if not rhs_cols:
    st.info("Select at least one regressor on the right-hand side to run the model.")
    st.stop()

# Build working frame (drop NA in raw inputs used)
needed_cols = set(rhs_cols) | {"__y__"}
if interact_emp_owner:
    if "emp_size" not in d.columns:
        st.error("`emp_size` is not available for interaction. (It should be created from `emp` earlier.)")
        st.stop()
    needed_cols |= {"emp_size", owner_var}

dwork = d.copy()
dwork["__y__"] = y
dwork = dwork[list(needed_cols)].replace([np.inf, -np.inf], np.nan).dropna()

if dwork.empty:
    st.error("No observations available after dropping missing values in outcome/regressors.")
    st.stop()


# ------------------------------------------------------
# Design matrix
# ------------------------------------------------------
X_parts = []

# Continuous vars (+ optional square per variable + log)
for v in cont_vars:
    x = pd.to_numeric(dwork[v], errors="coerce").astype(float)
    X_parts.append(x.rename(v))

    if v in quad_set:
        X_parts.append((x**2).rename(f"{v}^2"))

    if v in log_set:
        x_pos = x.where(x > 0, np.nan)         # nonpositive -> NA for log
        X_parts.append(np.log(x_pos).rename(f"log({v})"))

# Categorical vars -> one-hot (numeric dummies)
dummy_cache = {}  # store for interaction use
for v in cat_vars:
    dummies = pd.get_dummies(
        dwork[v].astype("category"),
        prefix=v,
        drop_first=True,
        dtype=float
    )
    dummy_cache[v] = dummies
    X_parts.append(dummies)

# Optional interaction: emp_size × ownership
if interact_emp_owner:
    if "emp_size" not in dummy_cache:
        emp_dum = pd.get_dummies(
            dwork["emp_size"].astype("category"),
            prefix="emp_size",
            drop_first=True,
            dtype=float
        )
    else:
        emp_dum = dummy_cache["emp_size"]

    own_dum = pd.get_dummies(
        dwork[owner_var].astype("category"),
        prefix=owner_var,
        drop_first=True,
        dtype=float
    )

    inter_cols = {}
    # pairwise product of columns
    for e_name in emp_dum.columns:
        for o_name in own_dum.columns:
            inter = (emp_dum[e_name] * own_dum[o_name]).astype(float)
            inter_cols[f"{e_name} × {o_name}"] = inter
    if inter_cols:
        X_parts.append(pd.DataFrame(inter_cols, index=dwork.index))

# Concatenate & coerce numeric, align with Y
X = pd.concat(X_parts, axis=1)
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
Y = pd.to_numeric(dwork["__y__"], errors="coerce").replace([np.inf, -np.inf], np.nan)

valid = X.notnull().all(axis=1) & Y.notnull()
X = X.loc[valid]
Y = Y.loc[valid]

if X.shape[0] < 5 or X.shape[1] == 0:
    st.error("Not enough usable data after encoding/coercion. Try different regressors or industry.")
    st.stop()

X = sm.add_constant(X, has_constant="add")
model = sm.OLS(Y.astype(float), X.astype(float))
res = model.fit(cov_type="HC1") 

# ------------------------------------------------------
# Pretty Output (stars + SE in parentheses)
# ------------------------------------------------------
st.subheader("Regression results")

def sig_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""

def fmt_coef(x):
    # 3 decimals by default; fall back to 3 sig figs if very small
    try:
        if x != 0 and abs(x) < 0.001:
            return f"{x:.3g}"
        return f"{x:.3f}"
    except Exception:
        return str(x)

def fmt_se(x):
    try:
        if x != 0 and abs(x) < 0.001:
            return f"{x:.3g}"
        return f"{x:.3f}"
    except Exception:
        return str(x)

def prettify_name(name: str) -> str:
    # Intercept
    if name == "const":
        return "Intercept"
    # Quadratic markers
    name = name.replace("^2", " squared").replace("^3", " cubed").replace("^4", " quartic")
    # Dummies like "gender_Female" -> "gender = Female"
    for cat in cat_vars:                         # <--- uses your selected categoricals
        prefix = f"{cat}_"
        if name.startswith(prefix):
            level = name[len(prefix):]
            return f"{cat} = {level}"
    # Generic cleanup
    return name.replace("_", " ")

# Build rows in the same order as the model params
rows = []
for term in res.params.index:
    coef = res.params[term]
    se   = res.bse[term]
    pval = res.pvalues[term]
    entry = f"{fmt_coef(coef)}{sig_stars(pval)} ({fmt_se(se)})"
    rows.append((prettify_name(term), entry))

# Create a one-model table
table_df = pd.DataFrame(rows, columns=["", "Model 1"])

# Render with Streamlit styling
st.table(table_df)
# Divider + bottom stats + baselines
st.markdown("---")
st.markdown(f"**R-squared**: {res.rsquared:.3f}")

note = "Robust standard errors (HC1) are in parentheses."

# --- Baselines (drop-first) for all categoricals used
def _baseline_of(series: pd.Series) -> str:
    s = series.astype("category")
    cats = list(s.cat.categories)
    return "—" if len(cats) == 0 else str(cats[0])

cats_in_model = set(cat_vars)
# Also include categoricals that appear only in interaction
if 'interact_emp_owner' in locals() and interact_emp_owner:
    cats_in_model |= {"emp_size", owner_var}

baseline_lines = []
for v in sorted(cats_in_model):
    if v in dwork.columns:
        baseline_lines.append(f"{v}: {_baseline_of(dwork[v])}")

baselines_text = " | ".join(baseline_lines) if baseline_lines else "None"

st.markdown(
    f"<span style='font-size:0.9em'><em>Notes:</em> {note} "
    "Significance levels: *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1.</span>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<span style='font-size:0.9em'><em>Category baselines (drop-first):</em> {baselines_text}</span>",
    unsafe_allow_html=True,
)
