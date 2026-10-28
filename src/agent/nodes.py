"""LangGraph node functions.

Design: the deterministic analytics tools do the math; the LLM reasons over the results
(prioritizing, deciding actions, explaining to a trader). In mock mode each node falls
back to a rule-based stand-in so the graph still produces sensible output with no key.
"""
from __future__ import annotations

import json

from src.agent.llm import complete, get_llm
from src.agent.state import AgentState
from src.tools import analytics

# one shared client per run (None in mock mode)
_LLM = None


def _llm():
    global _LLM
    if _LLM is None:
        _LLM = get_llm()
    return _LLM


# ---------------------------------------------------------------- ingest
def ingest(state: AgentState) -> AgentState:
    cid = state["campaign_id"]
    return {
        "kpis": analytics.campaign_kpis(cid),
        "underperformers": analytics.find_underperformers(cid),
        "opportunities": analytics.find_opportunities(cid),
        "reallocation": analytics.simulate_reallocation(cid),
    }


# ---------------------------------------------------------------- diagnose
def diagnose(state: AgentState) -> AgentState:
    o = state["kpis"]["overall"]
    unders = state["underperformers"]
    facts = (f"Campaign ROAS {o['ROAS']}, spend ₹{o['spend']:,.0f}, "
             f"revenue ₹{o['revenue']:,.0f}, {o['conversions']} conversions.\n"
             f"Flagged line items:\n{json.dumps(unders, indent=2)}")

    if state["mode"] == "claude":
        sys = ("You are a senior programmatic trading analyst at MiQ. In 3-4 sentences, "
               "diagnose what is dragging this campaign down. Be specific and quantitative; "
               "name the worst line items and the money at stake. No preamble.")
        return {"diagnosis": complete(_llm(), sys, facts)}

    # mock: deterministic diagnosis from the numbers
    if not unders:
        return {"diagnosis": f"Campaign is healthy at ROAS {o['ROAS']} — no money-losing line items."}
    total_waste = sum(u["wasted_spend"] for u in unders)
    worst = unders[0]
    return {"diagnosis": (
        f"Campaign ROAS is {o['ROAS']} but {len(unders)} line items are underperforming, "
        f"burning ~₹{total_waste:,.0f} of spend below break-even. The worst is {worst['line_item']} "
        f"({worst['channel']}): {worst['reasons']}, wasting ₹{worst['wasted_spend']:,.0f}.")}


# ---------------------------------------------------------------- recommend
def _rule_based_recs(state: AgentState) -> list[dict]:
    recs = []
    for u in state["underperformers"]:
        if u["ROAS"] < 0.5:
            recs.append({"action": "PAUSE", "line_item": u["line_item"],
                         "detail": f"ROAS {u['ROAS']} — stop the bleed",
                         "frees_spend": round(u["spend"], 0)})
        else:
            recs.append({"action": "CUT_BUDGET_50%", "line_item": u["line_item"],
                         "detail": f"ROAS {u['ROAS']} — halve budget, lower bids",
                         "frees_spend": round(u["spend"] * 0.5, 0)})
    opps = state["opportunities"]
    realloc = state["reallocation"]
    if opps:
        recs.append({"action": "SCALE_UP", "line_item": opps[0]["line_item"],
                     "detail": f"Top ROAS {opps[0]['ROAS']} — absorb freed budget",
                     "expected_uplift_inr": realloc.get("uplift", 0)})
    return recs


def recommend(state: AgentState) -> AgentState:
    if state["mode"] == "claude":
        sys = ("You are a MiQ trading agent. Given the campaign data, output ONLY a JSON array "
               "of concrete optimization actions. Each item: "
               '{"action","line_item","detail","frees_spend" or "expected_uplift_inr"}. '
               "Actions ∈ PAUSE, CUT_BUDGET_50%, LOWER_BID, SCALE_UP. Prioritize the biggest "
               "wasted spend first. No prose outside the JSON.")
        payload = json.dumps({"underperformers": state["underperformers"],
                              "opportunities": state["opportunities"],
                              "reallocation": state["reallocation"]}, indent=2)
        raw = complete(_llm(), sys, payload)
        try:
            start, end = raw.find("["), raw.rfind("]") + 1
            return {"recommendations": json.loads(raw[start:end])}
        except Exception:
            return {"recommendations": _rule_based_recs(state)}  # graceful fallback
    return {"recommendations": _rule_based_recs(state)}


# ---------------------------------------------------------------- report
def report(state: AgentState) -> AgentState:
    o = state["kpis"]["overall"]
    realloc = state["reallocation"]
    if state["mode"] == "claude":
        sys = ("You are a MiQ trading agent writing a crisp brief for a programmatic trader. "
               "Structure: 1) one-line headline, 2) what's wrong, 3) recommended actions as a "
               "bulleted list, 4) projected impact in ₹ and %. Be decisive and specific.")
        payload = json.dumps({"overall": o, "diagnosis": state["diagnosis"],
                              "recommendations": state["recommendations"],
                              "reallocation": realloc}, indent=2)
        return {"report": complete(_llm(), sys, payload)}

    # mock: assemble a trader-ready brief from the structured results
    lines = [f"CAMPAIGN OPTIMIZATION BRIEF — {state['campaign_id']}",
             f"Headline: ROAS {o['ROAS']} on ₹{o['spend']:,.0f} spend; reallocation can lift "
             f"revenue ~{realloc.get('uplift_pct', 0)}% (₹{realloc.get('uplift', 0):,.0f}).",
             "", "What's wrong:", f"  {state['diagnosis']}", "", "Recommended actions:"]
    for r in state["recommendations"]:
        extra = (f" (frees ₹{r['frees_spend']:,.0f})" if "frees_spend" in r
                 else f" (uplift ₹{r.get('expected_uplift_inr', 0):,.0f})" if "expected_uplift_inr" in r else "")
        lines.append(f"  • {r['action']} {r['line_item']} — {r['detail']}{extra}")
    lines += ["", "Projected impact:",
              f"  Move ₹{realloc.get('moved_spend', 0):,.0f} from {len(realloc.get('from', []))} "
              f"weak line items into {realloc.get('to', 'n/a')} (ROAS {realloc.get('to_ROAS', 'n/a')}).",
              f"  Revenue ₹{realloc.get('revenue_before', 0):,.0f} → ₹{realloc.get('revenue_after', 0):,.0f} "
              f"(+{realloc.get('uplift_pct', 0)}%)."]
    return {"report": "\n".join(lines)}


# ---------------------------------------------------------------- routing
def has_issues(state: AgentState) -> str:
    """Conditional edge: only recommend actions if we actually found problems."""
    return "recommend" if state["underperformers"] else "report"
