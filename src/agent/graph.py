"""The LangGraph: ingest → diagnose → (conditional) → recommend → report.

A stateful graph, not a single prompt — the conditional edge means the agent only
spends tokens recommending fixes when it actually found problems. This is the pattern
behind MiQ's Sigma 'Trading Agent': fragmented signals in, a decision + brief out.

    ingest ──▶ diagnose ──▶ [has issues?] ──yes──▶ recommend ──▶ report ──▶ END
                                   └──────no────────────────────▶ report
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent import nodes
from src.agent.state import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("ingest", nodes.ingest)
    g.add_node("diagnose", nodes.diagnose)
    g.add_node("recommend", nodes.recommend)
    g.add_node("report", nodes.report)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "diagnose")
    g.add_conditional_edges("diagnose", nodes.has_issues,
                            {"recommend": "recommend", "report": "report"})
    g.add_edge("recommend", "report")
    g.add_edge("report", END)
    return g.compile()
