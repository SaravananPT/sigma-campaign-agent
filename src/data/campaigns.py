"""Synthetic programmatic-advertising campaign data.

Models what a MiQ trader actually sees: one campaign split into line items across
channels (CTV, display, online video, native), with daily spend / impressions /
clicks / conversions / revenue. We deliberately plant a few underperformers (money-
losing segments) and a star performer so the optimization agent has real problems to
find — mirroring the fragmented, messy signal MiQ's Sigma platform is built to untangle.

Swap this for a real DSP export (same columns) and the agent is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

CHANNELS = ["CTV", "Display", "Online Video", "Native"]

# line_item -> (channel, base CPM ₹, base CTR, base CVR, avg order value ₹, label)
# labels let us verify the agent catches what we planted.
LINE_ITEMS = {
    "LI_01_CTV_Sports":      ("CTV",          350, 0.009, 0.040, 2400, "healthy"),
    "LI_02_CTV_News":        ("CTV",          420, 0.006, 0.012, 2200, "underperformer"),  # pricey CTV, weak CVR
    "LI_03_Display_Retarget":("Display",       90, 0.004, 0.030, 1800, "star"),            # cheap + converts
    "LI_04_Display_Prospect":("Display",       75, 0.0015,0.004, 1500, "underperformer"),  # bleeds budget
    "LI_05_OLV_YouTube":     ("Online Video", 180, 0.011, 0.022, 2000, "healthy"),
    "LI_06_OLV_Premium":     ("Online Video", 260, 0.008, 0.016, 2100, "healthy"),
    "LI_07_Native_Discovery":("Native",       110, 0.006, 0.018, 1700, "healthy"),
    "LI_08_Native_Feed":     ("Native",        95, 0.003, 0.006, 1600, "underperformer"),  # low CVR feed
}


def generate(campaign_id: str = "CMP_Q4_RETAIL", days: int = 14, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-10-01", periods=days, freq="D")
    rows = []
    for li, (channel, cpm, ctr, cvr, aov, label) in LINE_ITEMS.items():
        # daily budget pressure differs per line item
        daily_imps = rng.integers(60_000, 180_000)
        for d in dates:
            imps = int(daily_imps * rng.uniform(0.8, 1.2))
            clicks = rng.binomial(imps, ctr)
            conversions = rng.binomial(clicks, cvr) if clicks else 0
            spend = imps / 1000 * cpm * rng.uniform(0.95, 1.05)
            revenue = conversions * aov * rng.uniform(0.9, 1.1)
            rows.append({
                "date": d, "campaign_id": campaign_id, "line_item": li,
                "channel": channel, "impressions": imps, "clicks": int(clicks),
                "conversions": int(conversions), "spend": round(spend, 2),
                "revenue": round(revenue, 2), "_label": label,
            })
    return pd.DataFrame(rows)


def load(campaign_id: str = "CMP_Q4_RETAIL") -> pd.DataFrame:
    """Load the campaign, generating + caching it on first use."""
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{campaign_id}.parquet"
    if not path.exists():
        generate(campaign_id).to_parquet(path, index=False)
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = generate()
    DATA.mkdir(exist_ok=True)
    df.to_parquet(DATA / "CMP_Q4_RETAIL.parquet", index=False)
    print(f"[data] {len(df)} rows, {df['line_item'].nunique()} line items, {df['date'].nunique()} days")
    print(df.groupby("channel")[["spend", "revenue", "conversions"]].sum().round(0).to_string())
