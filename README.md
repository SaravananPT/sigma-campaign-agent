# Sigma Campaign Optimization Agent

A LangGraph + Claude agent that reads a programmatic ad campaign, finds the line items
losing money, and writes a trader-ready brief with concrete budget-reallocation actions —
a working prototype of the kind of **automated optimization** MiQ's Sigma "Trading Agent"
performs.

> **Why this problem?** MiQ's own research says traders "spend excessive time assembling
> data from disparate inputs rather than developing strategy," and MiQ's $100M **Sigma**
> platform is built around AI agents (a Planning Agent, a Trading Agent that does automated
> optimization, and Total Measurement) to fix programmatic's *fragmentation* and *speed of
> insight* problems. This agent tackles exactly that: fragmented signal in → decision out, in
> seconds.

## What it does

```
ingest ──▶ diagnose ──▶ [found issues?] ──yes──▶ recommend ──▶ report ──▶ END
                              └──────no──────────────────────▶ report
```

1. **Ingest** — pull campaign performance across line items, channels (CTV / display /
   online video / native), spend, impressions, clicks, conversions, revenue.
2. **Diagnose** — compute KPIs (CTR, CVR, CPM, **CPA**, **ROAS**) and flag line items that
   lose money (ROAS < 1.0) or convert far below average, ranked by wasted spend.
3. **Recommend** — decide concrete actions: PAUSE, cut budget, lower bid, or SCALE_UP.
4. **Report** — write a crisp brief for a trader, with projected ₹ and % impact.

The **conditional edge** means the agent only spends tokens recommending fixes when it
actually found problems — a small but real piece of agent design.

## Sample output (mock mode, free)

```
Headline: ROAS 1.51 on ₹3,042,729 spend; reallocation can lift revenue ~38.5% (₹1,774,873).
Recommended actions:
  • PAUSE LI_02_CTV_News — ROAS 0.29 — stop the bleed (frees ₹786,697)
  • PAUSE LI_04_Display_Prospect — ROAS 0.21 — stop the bleed (frees ₹153,465)
  • SCALE_UP LI_05_OLV_YouTube — Top ROAS 2.72 — absorb freed budget (uplift ₹1,774,873)
```

## Run it

```bash
pip install -r requirements.txt
python run.py                     # runs FREE in mock mode
export ANTHROPIC_API_KEY=sk-...   # then Claude (claude-opus-4-8) writes the reasoning
python run.py
pytest                            # unit tests for the analytics math
```

## Design

- **The LLM reasons; Python computes.** Deterministic analytics tools (`src/tools/`) do all
  the math so numbers are always correct and auditable; Claude prioritizes, decides actions,
  and explains. Swap the synthetic data (`src/data/`) for a real DSP export with the same
  columns and the agent is unchanged.
- **Mock fallback.** No API key? A rule-based stand-in runs the whole graph for free. Plug a
  key in for real Claude reasoning — same graph, same tools.
- **Claude via LangChain** — `ChatAnthropic(model="claude-opus-4-8")`, adaptive thinking on.

## Stack
Python · **LangGraph** · **LangChain** · **Claude (Anthropic)** · pandas · pytest

## Maps to the MiQ ML Engineer II role
Agentic LLM system (LLMOps) · production-shaped pipeline with a decision output · clean
tool/reasoning separation · unit-tested · built around MiQ's actual product direction.
