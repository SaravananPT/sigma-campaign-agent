"""Shared state passed between LangGraph nodes."""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    campaign_id: str
    mode: str                 # "claude" or "mock"
    kpis: dict                # overall + per-segment
    underperformers: list     # flagged money-losers
    opportunities: list       # scale-up candidates
    reallocation: dict        # simulated budget move
    diagnosis: str            # narrative: what's wrong
    recommendations: list     # concrete actions
    report: str               # final trader-ready brief
