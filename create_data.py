
#############
# This is a full-random simulation, without any input from the real data, should not use this
#############

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict

# =====================================================
# Synthetic data generator for firm-level datasets
# - Cross-section (2019): ~100k firms across industries
# - Panel (2010–2022): food production industry (~20k firms/year)
#
# All monetary variables are in 1000 HUF units, as requested.
# =====================================================

SEED = 42
rng = np.random.default_rng(SEED)

# -------------------------------
# Helpers
# -------------------------------

def _bounded_normal(mean: float, sd: float, size: int, low: float, high: float) -> np.ndarray:
    """Draw from a normal distribution clipped to [low, high]."""
    x = rng.normal(mean, sd, size)
    return np.clip(x, low, high)


def _zero_inflated_lognormal(size: int, zero_prob: float, mean: float, sigma: float) -> np.ndarray:
    """Return zero-inflated lognormal draws."""
    is_zero = rng.random(size) < zero_prob
    draws = rng.lognormal(mean=mean, sigma=sigma, size=size)
    draws[is_zero] = 0.0
    return draws


def _sample_ownership(n: int) -> np.ndarray:
    cats = np.array(["domestic_private", "foreign_private", "public"])
    probs = np.array([0.77, 0.18, 0.05])  # rough stylized split
    return rng.choice(cats, size=n, p=probs)


def _sample_industries(n: int) -> np.ndarray:
    """Sample 2-digit industry codes (as strings) with a skew toward common sectors."""
    # A small, realistic set of common NACE-like 2-digit codes for illustration
    inds = np.array(["10", "11", "20", "23", "25", "28", "33", "41", "43", "45",
                     "46", "47", "49", "52", "55", "56", "62", "68"])
    probs = np.array([0.07, 0.02, 0.05, 0.03, 0.06, 0.05, 0.03, 0.05, 0.05, 0.05,
                      0.11, 0.13, 0.04, 0.04, 0.04, 0.07, 0.06, 0.10])
    probs = probs / probs.sum()
    return rng.choice(inds, size=n, p=probs)


# -------------------------------
# Cross-section (2019)
# -------------------------------

def make_cross_section_2019(n_firms: int = 100_000) -> pd.DataFrame:
    year = 2019

    firm_id = np.array([f"C{year}_{i:06d}" for i in range(1, n_firms + 1)])
    industry = _sample_industries(n_firms)
    ownership = _sample_ownership(n_firms)

    # Employment: discrete lognormal, at least 1 employee
    emp = np.rint(rng.lognormal(mean=2.0, sigma=1.0, size=n_firms)).astype(int)
    emp = np.maximum(emp, 1)

    # Sales per employee (1000 HUF): lognormal; larger firms tend to have a bit higher productivity
    base_spe = rng.lognormal(mean=6.5, sigma=0.7, size=n_firms)  # median ~ 665 in 1000 HUF
    size_boost = 1.0 + 0.15 * np.log1p(emp)  # modest uplift for larger firms
    sales = base_spe * emp * size_boost

    # Tangible assets correlate with sales; ratio is lognormal, with capital intensity by industry
    cap_ratio = rng.lognormal(mean=-0.8, sigma=0.5, size=n_firms)  # typically < sales
    # Industry adjustment (e.g., manufacturing more capital intensive than services)
    manuf_like = np.isin(industry, ["10", "11", "20", "23", "25", "28", "33"])  # boolean mask
    cap_ratio *= np.where(manuf_like, 1.4, 0.8)
    tangible_assets = sales * cap_ratio

    # Export status depends on ownership and firm size
    p_export = 0.15 + 0.25 * (ownership == "foreign_private") + 0.15 * (emp >= 50) + 0.05 * (emp >= 250)
    p_export = np.clip(p_export, 0.02, 0.9)
    exporter = (rng.random(n_firms) < p_export).astype(int)

    # Profit margin: clipped normal; allow losses
    pm = _bounded_normal(mean=0.07, sd=0.10, size=n_firms, low=-0.5, high=0.5)
    profit = sales * pm

    # Grants: zero-inflated lognormal; bigger firms more likely to receive some support
    grant_zero_prob = np.clip(0.90 - 0.10 * (emp >= 50) - 0.15 * (ownership == "public"), 0.50, 0.95)
    grant = _zero_inflated_lognormal(size=n_firms, zero_prob=grant_zero_prob, mean=7.0, sigma=1.0)

    df = pd.DataFrame({
        "firm_id": firm_id,
        "year": year,
        "industry_code": industry.astype(str),
        "sales_1000HUF": sales.astype(float),
        "employment": emp.astype(int),
        "tangible_assets_1000HUF": tangible_assets.astype(float),
        "exporter": exporter.astype(int),
        "grant_1000HUF": grant.astype(float),
        "profit_1000HUF": profit.astype(float),
        "ownership": ownership.astype(str),
    })

    # Enforce types & reasonable floors
    for col in ["sales_1000HUF", "tangible_assets_1000HUF", "grant_1000HUF"]:
        df[col] = np.maximum(df[col], 0.0)

    return df


# -------------------------------
# Panel (2010–2022) for food production (industry code '10')
# -------------------------------

@dataclass
class PanelConfig:
    start_year: int = 2010
    end_year: int = 2022
    target_per_year: int = 20_000
    entrant_share: float = 0.12   # of target, rough inflow each year
    exit_prob: float = 0.10       # annual exit probability per firm


def _make_food_panel_ids(cfg: PanelConfig) -> Dict[int, np.ndarray]:
    years = np.arange(cfg.start_year, cfg.end_year + 1)

    # Create a pool large enough to allow churn while keeping ~20k active/year
    pool_size = int(cfg.target_per_year * 1.6)
    pool_ids = np.array([f"F{y}{i:06d}" for i, y in enumerate(rng.choice(years, size=pool_size, replace=True), start=1)])

    active_by_year: Dict[int, np.ndarray] = {}

    # Initialize active set for first year
    n0 = int(cfg.target_per_year * rng.uniform(0.9, 1.05))
    active = rng.choice(pool_ids, size=n0, replace=False)

    for t, y in enumerate(years):
        if t > 0:
            # Survivors
            survive_mask = rng.random(active.size) >= cfg.exit_prob
            survivors = active[survive_mask]

            # Entrants to hit target +/- small noise
            target = int(cfg.target_per_year * rng.uniform(0.95, 1.05))
            need = max(target - survivors.size, 0)
            entrants = rng.choice(np.setdiff1d(pool_ids, survivors, assume_unique=False), size=need, replace=False) if need > 0 else np.array([], dtype=pool_ids.dtype)
            active = np.concatenate([survivors, entrants])

        active_by_year[y] = np.sort(active)

    return active_by_year


def make_food_panel(cfg: PanelConfig = PanelConfig()) -> pd.DataFrame:
    industry_code = "10"  # food production
    years = np.arange(cfg.start_year, cfg.end_year + 1)

    active_ids_by_year = _make_food_panel_ids(cfg)

    # Firm-level persistent traits
    all_ids = np.unique(np.concatenate(list(active_ids_by_year.values())))
    n_pool = all_ids.size

    ownership = dict(zip(all_ids, _sample_ownership(n_pool)))

    # Firm fixed effects (productivity / capital intensity)
    firm_size_fe = dict(zip(all_ids, rng.normal(0.0, 0.5, size=n_pool)))
    firm_capint_fe = dict(zip(all_ids, rng.normal(0.0, 0.3, size=n_pool)))

    rows = []

    # Nominal growth trend and shocks
    # Assume modest nominal growth in sales per employee and assets
    for y in years:
        ids = active_ids_by_year[y]
        n = ids.size

        # Time trends
        t = y - years[0]
        trend_sales = np.exp(0.03 * t)   # ~3% growth per year
        trend_assets = np.exp(0.025 * t) # ~2.5% growth per year

        # Base employment from a lognormal with firm FE persistence
        base_emp = rng.lognormal(mean=2.0 + 0.15 * np.array([firm_size_fe[i] for i in ids]), sigma=0.9, size=n)
        emp = np.maximum(np.rint(base_emp).astype(int), 1)

        # Sales per employee with growth trend + idiosyncratic shock
        spe = rng.lognormal(mean=6.4 + 0.10 * np.array([firm_size_fe[i] for i in ids]), sigma=0.6, size=n) * trend_sales
        sales = spe * emp

        # Tangible assets scale with sales and capital-intensity FE
        cap_ratio = rng.lognormal(mean=-0.7 + 0.15 * np.array([firm_capint_fe[i] for i in ids]), sigma=0.45, size=n) * trend_assets
        tangible_assets = sales * cap_ratio

        # Export status depends on ownership, size, and a mild upward trend
        own = np.array([ownership[i] for i in ids])
        p_export = 0.12 + 0.02 * t + 0.20 * (own == "foreign_private") + 0.12 * (emp >= 50) + 0.05 * (emp >= 250)
        p_export = np.clip(p_export, 0.02, 0.9)
        exporter = (rng.random(n) < p_export).astype(int)

        # Profit margin with small business cycle variation
        cycle = 0.01 * np.sin(2 * np.pi * (t / 6.0))  # gentle cycles
        pm = _bounded_normal(mean=0.06 + cycle, sd=0.09, size=n, low=-0.5, high=0.5)
        profit = sales * pm

        # Grants: zero-inflated; public firms & larger firms more likely, some year-to-year variation
        grant_zero_prob = np.clip(0.92 - 0.12 * (emp >= 50) - 0.18 * (own == "public") + 0.02 * rng.standard_normal(n), 0.50, 0.97)
        grant = _zero_inflated_lognormal(size=n, zero_prob=grant_zero_prob, mean=6.8 + 0.015 * t, sigma=1.0)

        rows.append(pd.DataFrame({
            "firm_id": ids,
            "year": y,
            "industry_code": industry_code,
            "sales_1000HUF": np.maximum(sales, 0.0),
            "employment": emp,
            "tangible_assets_1000HUF": np.maximum(tangible_assets, 0.0),
            "exporter": exporter,
            "grant_1000HUF": np.maximum(grant, 0.0),
            "profit_1000HUF": profit,
            "ownership": own,
        }))

    panel_df = pd.concat(rows, ignore_index=True)
    return panel_df


# -------------------------------
# Entry point to generate and persist datasets
# -------------------------------

def build_datasets(
    n_cross_section: int = 100_000,
    panel_cfg: PanelConfig = PanelConfig(),
    save_parquet: bool = True,
    out_dir: str = "data/synthetic"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import os
    os.makedirs(out_dir, exist_ok=True)

    cross_df = make_cross_section_2019(n_cross_section)
    panel_df = make_food_panel(panel_cfg)

    if save_parquet:
        cross_path = f"{out_dir}/cross_section_2019.parquet"
        panel_path = f"{out_dir}/panel_food_2010_2022_easy_simul.parquet"
        cross_df.to_parquet(cross_path, index=False)
        panel_df.to_parquet(panel_path, index=False)
        print(f"Saved: {cross_path} ({len(cross_df):,} rows)")
        print(f"Saved: {panel_path} ({len(panel_df):,} rows)")

    # Quick sanity output
    print("Cross-section 2019 — head()\n", cross_df.head())
    print("\nPanel — years coverage:", sorted(panel_df['year'].unique()))
    print(panel_df.groupby('year').size().describe())

    return cross_df, panel_df


if __name__ == "__main__":
    # Generate datasets and save to ./data as parquet files
    build_datasets()
