# SAR Processing System — Multi-Agent Financial Compliance

This project implements an **AI-powered Suspicious Activity Report (SAR) processing system** for financial crime detection. It uses a two-agent architecture with human-in-the-loop review and full audit trails.

## Overview

The system automates the workflow that financial institutions use to detect and report suspicious activity to FinCEN:

1. **Risk screening** — AI analyzes transaction patterns and classifies suspicious activity.
2. **Human review** — Compliance officers approve or reject before narrative generation.
3. **Narrative generation** — Approved cases get regulatory-compliant SAR narratives.
4. **SAR filing** — Complete forms and audit logs are produced for examination.

### Architecture

```
CSV Data → Risk Analyst Agent → Human Review → Compliance Officer Agent → SAR Documents
            (Chain-of-Thought)      (Gate)         (ReACT Framework)
```

- **Risk Analyst Agent** — Chain-of-Thought reasoning; classifies into Structuring, Sanctions, Fraud, Money Laundering, Other.
- **Compliance Officer Agent** — ReACT-style prompting; generates ≤120-word narratives with regulatory citations.
- **Human-in-the-loop** — Decision gates for compliance; every step is logged.

### Why SAR Processing Matters

- Institutions must file SARs within 30 days of detection; non-compliance carries severe penalties.
- This system addresses volume, consistency, cost, and auditability using a two-stage workflow that limits API use and keeps narratives compliant.

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (or any OpenAI-compatible API endpoint)

### Setup

```bash
cd starter
pip install -r requirements.txt
cp .env.template .env
# Edit .env and set OPENAI_API_KEY=your-key-here
```

For an alternate base URL (e.g. proxy or different provider), set `OPENAI_API_KEY` and, if needed, configure the client in code to use a different `base_url`.

### Project Structure

```
starter/
├── README.md
├── requirements.txt
├── .env.template
├── data/
│   ├── customers.csv
│   ├── accounts.csv
│   └── transactions.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_agent_development.ipynb
│   └── 03_workflow_integration.ipynb
├── src/
│   ├── foundation_sar.py       # Data schemas, loader, audit logging
│   ├── risk_analyst_agent.py   # Chain-of-Thought risk classification
│   └── compliance_officer_agent.py  # ReACT narrative generation
├── tests/
│   ├── test_foundation.py
│   ├── test_risk_analyst.py
│   └── test_compliance_officer.py
├── outputs/
│   ├── filed_sars/
│   └── audit_logs/
└── docs/
    ├── system_architecture.md
    ├── prompting_guide.md
    ├── regulatory_context.md
    └── troubleshooting.md
```

## Running the System

### Tests

```bash
# Run all tests
python -m pytest tests/ -v

# By component
python -m pytest tests/test_foundation.py -v
python -m pytest tests/test_risk_analyst.py -v
python -m pytest tests/test_compliance_officer.py -v
```

There are 30 tests across foundation (schemas, loader, audit), risk analyst (API integration, parsing, errors), and compliance officer (narrative length, citations, validation).

### End-to-End Workflow

Use **`notebooks/03_workflow_integration.ipynb`** to run the full pipeline from CSV data through risk analysis, human review, and SAR generation.

## Implementation Highlights

- **Pydantic** — Typed schemas for customers, accounts, transactions, and cases; validation and error handling.
- **Chain-of-Thought** — Structured prompts for step-by-step risk analysis and classification.
- **ReACT** — Reasoning + action flow for compliant narrative generation and citations.
- **Audit trails** — All decisions and agent outputs logged for regulatory review.

## Data

Sample data under `data/` includes customers, accounts, and transactions with patterns relevant to structuring, money laundering, sanctions, and fraud. The pipeline loads and aggregates this into cases for the agents.

## Documentation

- **`docs/system_architecture.md`** — End-to-end data flow and component roles.
- **`docs/prompting_guide.md`** — Chain-of-Thought and ReACT usage.
- **`docs/regulatory_context.md`** — SAR/BSA context and requirements.

## License

See repository license file.
