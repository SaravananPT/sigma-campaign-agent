"""Tests for the deterministic analytics — the agent's math must be correct."""
import pandas as pd

from src.tools.analytics import _kpis, find_underperformers, simulate_reallocation


def test_kpis_math():
    df = pd.DataFrame([
        {"impressions": 1000, "clicks": 10, "conversions": 2, "spend": 100.0, "revenue": 300.0},
    ])
    k = _kpis(df)
    assert k["CTR"] == 0.01           # 10/1000
    assert k["CVR"] == 0.2            # 2/10
    assert k["CPM"] == 100.0          # 100/1000*1000
    assert k["CPA"] == 50.0           # 100/2
    assert k["ROAS"] == 3.0           # 300/100


def test_kpis_handles_zero_conversions():
    df = pd.DataFrame([{"impressions": 500, "clicks": 3, "conversions": 0, "spend": 50.0, "revenue": 0.0}])
    k = _kpis(df)
    assert k["CPA"] is None           # no divide-by-zero
    assert k["ROAS"] == 0.0


def test_finds_the_planted_money_loser():
    unders = find_underperformers("CMP_Q4_RETAIL")
    names = [u["line_item"] for u in unders]
    # the generator plants LI_02_CTV_News as a money-losing underperformer
    assert "LI_02_CTV_News" in names
    # ranked by wasted spend — the biggest bleed is first
    assert unders[0]["wasted_spend"] >= unders[-1]["wasted_spend"]
    assert all(u["ROAS"] < 1.0 or u["CPA"] for u in unders)


def test_reallocation_projects_positive_uplift():
    r = simulate_reallocation("CMP_Q4_RETAIL")
    # moving budget from ROAS<1 losers into the best performer must raise revenue
    assert r["uplift"] > 0
    assert r["revenue_after"] > r["revenue_before"]
