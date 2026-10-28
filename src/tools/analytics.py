"""Deterministic campaign analytics — the agent's 'hands'.

The LLM reasons and decides; these functions do the exact math so the numbers are
always correct and auditable. Exposed as LangChain tools so a tool-calling agent could
invoke them directly, and also callable in-process by the graph nodes.

KPIs (programmatic advertising):
  CTR  = clicks / impressions        CVR  = conversions / clicks
  CPM  = spend / impressions * 1000  CPA  = spend / conversions   (cost per acquisition)
  ROAS = revenue / spend             (return on ad spend; < 1.0 = losing money)
"""
from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from src.data.campaigns import load

GROUP = "line_item"


def _kpis(df: pd.DataFrame) -> dict:
    imp, clk = df["impressions"].sum(), df["clicks"].sum()
    conv, spend, rev = df["conversions"].sum(), df["spend"].sum(), df["revenue"].sum()
    return {
        "impressions": int(imp), "clicks": int(clk), "conversions": int(conv),
        "spend": round(float(spend), 0), "revenue": round(float(rev), 0),
        "CTR": round(clk / imp, 4) if imp else 0.0,
        "CVR": round(conv / clk, 4) if clk else 0.0,
        "CPM": round(spend / imp * 1000, 1) if imp else 0.0,
        "CPA": round(spend / conv, 0) if conv else None,
        "ROAS": round(rev / spend, 2) if spend else 0.0,
    }


def campaign_kpis(campaign_id: str = "CMP_Q4_RETAIL") -> dict:
    """Overall KPIs for the whole campaign, plus a per-line-item breakdown."""
    df = load(campaign_id)
    overall = _kpis(df)
    segments = {}
    for li, g in df.groupby(GROUP):
        k = _kpis(g)
        k["channel"] = g["channel"].iloc[0]
        segments[li] = k
    return {"campaign_id": campaign_id, "overall": overall, "segments": segments}


def find_underperformers(campaign_id: str = "CMP_Q4_RETAIL") -> list[dict]:
    """Flag line items that lose money (ROAS < 1.0) or convert far worse than the
    campaign average, ranked by wasted spend so the biggest bleed is first."""
    k = campaign_kpis(campaign_id)
    camp_cpa = k["overall"]["CPA"] or 1e9
    flagged = []
    for li, s in k["segments"].items():
        roas, cpa, spend = s["ROAS"], s["CPA"], s["spend"]
        loses_money = roas < 1.0
        bad_cpa = cpa is not None and cpa > 1.5 * camp_cpa
        if loses_money or bad_cpa:
            # wasted spend = the portion of spend that returned < ₹1 per ₹1
            wasted = round(spend * max(0.0, 1.0 - roas), 0)
            reasons = []
            if loses_money:
                reasons.append(f"ROAS {roas} (<1.0 = losing money)")
            if bad_cpa:
                reasons.append(f"CPA ₹{cpa:.0f} vs campaign ₹{camp_cpa:.0f}")
            flagged.append({
                "line_item": li, "channel": s["channel"], "spend": spend,
                "ROAS": roas, "CPA": cpa, "wasted_spend": wasted,
                "reasons": "; ".join(reasons),
            })
    return sorted(flagged, key=lambda x: x["wasted_spend"], reverse=True)


def find_opportunities(campaign_id: str = "CMP_Q4_RETAIL") -> list[dict]:
    """Line items worth MORE budget: strong ROAS, room to scale."""
    k = campaign_kpis(campaign_id)
    camp_roas = k["overall"]["ROAS"]
    opps = []
    for li, s in k["segments"].items():
        if s["ROAS"] >= max(1.5, 1.3 * camp_roas):
            opps.append({"line_item": li, "channel": s["channel"],
                         "ROAS": s["ROAS"], "spend": s["spend"]})
    return sorted(opps, key=lambda x: x["ROAS"], reverse=True)


def simulate_reallocation(campaign_id: str = "CMP_Q4_RETAIL", move_fraction: float = 0.7) -> dict:
    """What if we move `move_fraction` of each underperformer's spend into the best
    ROAS line items? Estimate incremental revenue at the receivers' efficiency."""
    k = campaign_kpis(campaign_id)
    unders = find_underperformers(campaign_id)
    opps = find_opportunities(campaign_id)
    if not unders or not opps:
        return {"moved_spend": 0, "revenue_before": k["overall"]["revenue"],
                "revenue_after": k["overall"]["revenue"], "uplift": 0, "note": "nothing to move"}

    moved = round(sum(u["spend"] * move_fraction for u in unders), 0)
    # revenue those underperformers were generating on the moved budget (lost by pausing)
    lost = round(sum(u["spend"] * move_fraction * u["ROAS"] for u in unders), 0)
    # revenue the moved budget earns at the top opportunity's ROAS
    best_roas = opps[0]["ROAS"]
    gained = round(moved * best_roas, 0)
    before = k["overall"]["revenue"]
    after = round(before - lost + gained, 0)
    return {
        "moved_spend": moved, "from": [u["line_item"] for u in unders],
        "to": opps[0]["line_item"], "to_ROAS": best_roas,
        "revenue_before": before, "revenue_after": after,
        "uplift": round(after - before, 0),
        "uplift_pct": round((after - before) / before * 100, 1) if before else 0.0,
    }


# --- LangChain tool wrappers (so a tool-calling agent can invoke these too) ---
campaign_kpis_tool = tool(campaign_kpis)
find_underperformers_tool = tool(find_underperformers)
find_opportunities_tool = tool(find_opportunities)
simulate_reallocation_tool = tool(simulate_reallocation)

TOOLS = [campaign_kpis_tool, find_underperformers_tool,
         find_opportunities_tool, simulate_reallocation_tool]


if __name__ == "__main__":
    import json
    print("OVERALL:", json.dumps(campaign_kpis()["overall"], indent=2))
    print("\nUNDERPERFORMERS:", json.dumps(find_underperformers(), indent=2))
    print("\nREALLOCATION:", json.dumps(simulate_reallocation(), indent=2))
