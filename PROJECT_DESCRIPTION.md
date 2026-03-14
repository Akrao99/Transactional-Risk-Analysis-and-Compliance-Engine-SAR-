# SAR Processing System — Project Overview

AI-powered **Suspicious Activity Report (SAR) processing system** for financial crime detection using a multi-agent architecture with Chain-of-Thought and ReACT prompting.

## What It Does

- **Risk Analyst Agent** — Chain-of-Thought reasoning to classify suspicious activity (Structuring, Sanctions, Fraud, Money Laundering, Other).
- **Compliance Officer Agent** — ReACT framework to generate regulatory-compliant SAR narratives (≤120 words, with citations).
- **End-to-end workflow** — CSV data → risk analysis → human review gate → narrative generation → SAR documents and audit trails.

## Architecture

```
CSV Data → Risk Analyst Agent → Human Review → Compliance Officer Agent → SAR Filing
            (Chain-of-Thought)      (Gate)         (ReACT Framework)
```

## Repository Layout

- **`starter/`** — Main project
  - `src/` — Core modules (foundation_sar, risk_analyst_agent, compliance_officer_agent)
  - `notebooks/` — 01_data_exploration, 02_agent_development, 03_workflow_integration
  - `tests/` — Unit tests for foundation and agents
  - `data/` — Sample customer, account, and transaction CSVs
  - `README.md` — Setup, usage, and testing

## Tech Stack

- Python 3.8+, Pydantic, OpenAI API, Jupyter, pytest, pandas
- Chain-of-Thought and ReACT prompting; human-in-the-loop; audit logging

## Getting Started

See **[README.md](README.md)** and **[starter/README.md](starter/README.md)** for setup, dependencies, and how to run tests and the workflow.

## Note

This project uses synthetic financial data and simulates regulatory workflows for demonstration purposes.
