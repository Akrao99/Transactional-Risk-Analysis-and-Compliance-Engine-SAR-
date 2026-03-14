# SAR Processing System — Multi-Agent Financial Compliance

An **AI-powered Suspicious Activity Report (SAR) processing system** that automates financial crime detection using a multi-agent architecture with Chain-of-Thought and ReACT prompting.

## What It Does

- **Risk Analyst Agent** — Classifies suspicious activity (structuring, sanctions, fraud, money laundering) using Chain-of-Thought reasoning.
- **Compliance Officer Agent** — Generates regulatory-compliant SAR narratives (≤120 words, with citations) using a ReACT-style framework.
- **End-to-end workflow** — CSV data → risk analysis → human review gate → narrative generation → SAR documents and audit trails.

## Tech Stack

- Python 3.8+, Pydantic, OpenAI API, Jupyter  
- Multi-agent design, human-in-the-loop, audit logging

## Quick Start

```bash
cd starter
pip install -r requirements.txt
cp .env.template .env   # Add your OPENAI_API_KEY
python -m pytest tests/ -v
```

See **[starter/README.md](starter/README.md)** for full setup, architecture, and usage.

## Repository Layout

- **`starter/`** — Main project: `src/`, `notebooks/`, `tests/`, `data/`, `docs/`
