"""Run the Sigma Campaign Optimization Agent.

    python run.py                 # analyze the default campaign
    python run.py CMP_Q4_RETAIL   # a specific campaign

Runs free in mock mode. Set ANTHROPIC_API_KEY to have Claude do the reasoning.
"""
from __future__ import annotations

import sys

from src.agent.graph import build_graph
from src.agent.llm import get_mode


def main() -> None:
    campaign_id = sys.argv[1] if len(sys.argv) > 1 else "CMP_Q4_RETAIL"
    mode = get_mode()

    print(f"\n{'='*68}\n  SIGMA CAMPAIGN OPTIMIZATION AGENT   ·   reasoning engine: {mode.upper()}\n{'='*68}")

    graph = build_graph()
    final = graph.invoke({"campaign_id": campaign_id, "mode": mode})

    o = final["kpis"]["overall"]
    print(f"\n[campaign] {campaign_id}  ·  ROAS {o['ROAS']}  ·  spend ₹{o['spend']:,.0f}"
          f"  ·  {o['conversions']} conversions")
    print(f"[flagged]  {len(final['underperformers'])} underperforming line items; "
          f"{len(final['opportunities'])} scale-up opportunities\n")

    print("─" * 68)
    print(final["report"])
    print("─" * 68)

    if mode == "mock":
        print("\n(ℹ︎ mock reasoning — set ANTHROPIC_API_KEY to have Claude write the brief)")


if __name__ == "__main__":
    main()
