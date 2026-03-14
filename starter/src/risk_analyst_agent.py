# risk_analyst_agent.py
import json
import openai
from openai import OpenAI
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
import os

from foundation_sar import (
    RiskAnalystOutput,
    ExplainabilityLogger,
    CaseData,
    CustomerData,
    AccountData,
    TransactionData
)

load_dotenv()


class RiskAnalystAgent:
    """Risk Analyst agent using Chain-of-Thought reasoning."""

    def __init__(self, openai_client, explainability_logger: ExplainabilityLogger, model="gpt-4o"):
        self.client = openai_client
        self.logger = explainability_logger
        self.model  = model

        self.system_prompt = """
You are a Senior Financial Crime Risk Analyst with 15+ years of experience in AML
compliance, fraud detection, and regulatory reporting. You specialize in identifying
suspicious activity patterns and filing SARs under BSA/AML regulations.

When analyzing a case, follow this exact 5-step Chain-of-Thought framework
(think step-by-step before reaching any conclusion):

STEP 1 - DATA REVIEW:
- Examine customer profile (identity, risk rating, occupation, income)
- Review account details (type, balance, account age, status)
- Catalog all transactions (amounts, types, methods, counterparties, dates)
- Note any missing or suspicious data points

STEP 2 - PATTERN RECOGNITION:
- Identify unusual transaction volumes or frequencies
- Look for structuring (transactions just below $10,000)
- Check for rapid movement of funds
- Identify unusual counterparties or locations
- Compare transactions against customer income and occupation

STEP 3 - REGULATORY MAPPING:
- BSA (31 CFR 1020.320) - SAR filing requirements
- FinCEN guidelines - suspicious activity indicators
- OFAC - sanctions screening
- 12 CFR 21.11 - national bank SAR requirements

STEP 4 - RISK QUANTIFICATION:
- Assign confidence score between 0.0 and 1.0
- Consider volume of suspicious indicators
- Consider customer risk rating and income vs transaction amounts
- Assign risk level: Low, Medium, High, or Critical

STEP 5 - CLASSIFICATION DECISION:
- Structuring: Transactions deliberately kept below $10,000
- Sanctions: Transactions involving OFAC-listed entities
- Fraud: Deceptive or identity-related suspicious activity
- Money_Laundering: Layering or integrating illicit funds
- Other: Suspicious activity not fitting above categories

Financial Crime Risk Analyst output must be valid JSON only.

CRITICAL: Respond with ONLY a valid JSON object. No explanation, no preamble,
no markdown. Just the raw JSON:
{
    "classification": "Structuring" | "Sanctions" | "Fraud" | "Money_Laundering" | "Other",
    "confidence_score": 0.0 to 1.0,
    "reasoning": "Step-by-step analysis (max 500 characters)",
    "key_indicators": ["indicator 1", "indicator 2"],
    "risk_level": "Low" | "Medium" | "High" | "Critical"
}
"""

    def analyze_case(self, case_data: CaseData) -> RiskAnalystOutput:
        """Perform risk analysis on a case using Chain-of-Thought reasoning."""
        start_time = datetime.now()

        try:
            user_prompt = self._format_case_for_prompt(case_data)

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_prompt}
                ]
            )

            response_content = response.choices[0].message.content

            try:
                json_str = self._extract_json_from_response(response_content)
                parsed   = json.loads(json_str)
                result   = RiskAnalystOutput(**parsed)
            except Exception as parse_error:
                raise ValueError(
                    f"Failed to parse Risk Analyst JSON output: {parse_error}"
                )

            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case_data.case_id,
                input_data={
                    "customer_id":       case_data.customer.customer_id,
                    "transaction_count": len(case_data.transactions),
                    "account_count":     len(case_data.accounts),
                },
                output_data={
                    "classification":   result.classification,
                    "confidence_score": result.confidence_score,
                    "risk_level":       result.risk_level,
                },
                reasoning=result.reasoning,
                execution_time_ms=execution_time_ms,
                success=True,
                error_message=None,
            )

            return result

        except Exception as e:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.log_agent_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case_data.case_id,
                input_data={"customer_id": case_data.customer.customer_id},
                output_data={},
                reasoning="JSON parsing failed",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e),
            )
            raise

    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON from LLM response handling all common wrapping patterns."""
        if not response_content or not response_content.strip():
            raise ValueError("No JSON content found in empty response")

        content = response_content.strip()

        # Pattern 1: ```json ... ```
        if "```json" in content:
            start = content.find("```json") + 7
            end   = content.find("```", start)
            if end != -1:
                return content[start:end].strip()

        # Pattern 2: plain ``` ... ```
        if "```" in content:
            start = content.find("```") + 3
            end   = content.find("```", start)
            if end != -1:
                return content[start:end].strip()

        # Pattern 3: raw JSON — find outermost { } pair
        start = content.find("{")
        end   = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return content[start:end + 1].strip()

        raise ValueError(f"No JSON content found in response: {content[:200]}")

    def _format_accounts(self, accounts: List[AccountData]) -> str:
        """Format account list for prompt injection."""
        if not accounts:
            return "  No accounts on record"
        return "\n".join([
            f"  {acc.account_id}: {acc.account_type} | "
            f"Balance: ${acc.current_balance:,.2f} | "
            f"Avg Monthly: ${acc.average_monthly_balance:,.2f} | "
            f"Status: {acc.status} | Opened: {acc.opening_date}"
            for acc in accounts
        ])

    def _format_transactions(self, transactions: List[TransactionData]) -> str:
        """Format transaction list for prompt injection."""
        if not transactions:
            return "  No transactions on record"
        lines = []
        for i, txn in enumerate(transactions, start=1):
            line = (
                f"{i}. {txn.transaction_date}: {txn.transaction_type} "
                f"${txn.amount:,.2f} via {txn.method}"
            )
            if txn.description:
                line += f" | {txn.description}"
            if txn.location:
                line += f" | {txn.location}"
            if txn.counterparty:
                line += f" | Counterparty: {txn.counterparty}"
            lines.append(line)
        return "\n".join(lines)

    def _format_case_for_prompt(self, case_data: CaseData) -> str:
        """Convert CaseData into a structured briefing document for the model."""
        amounts      = [txn.amount for txn in case_data.transactions]
        total_volume = sum(amounts)
        avg_amount   = total_volume / len(amounts) if amounts else 0
        below_10k    = [a for a in amounts if 9_000 <= a < 10_000]

        annual_income = (
            f"${case_data.customer.annual_income:,}"
            if case_data.customer.annual_income
            else "Not provided"
        )

        return f"""
CASE ID: {case_data.case_id}
CREATED: {case_data.case_created_at}

=== CUSTOMER PROFILE ===
ID:             {case_data.customer.customer_id}
Name:           {case_data.customer.name}
Date of Birth:  {case_data.customer.date_of_birth}
Address:        {case_data.customer.address}
Occupation:     {case_data.customer.occupation or 'Not provided'}
Annual Income:  {annual_income}
Risk Rating:    {case_data.customer.risk_rating}
Customer Since: {case_data.customer.customer_since}

=== ACCOUNT DETAILS ===
{self._format_accounts(case_data.accounts)}

=== TRANSACTIONS ({len(case_data.transactions)} total) ===
{self._format_transactions(case_data.transactions)}

=== FINANCIAL SUMMARY ===
Total Volume:                   ${total_volume:,.2f}
Average Transaction:            ${avg_amount:,.2f}
Transactions $9,000-$9,999:     {len(below_10k)} (structuring indicator)
Unique Transaction Types:       {len(set(t.transaction_type for t in case_data.transactions))}
Counterparties Identified:      {sum(1 for t in case_data.transactions if t.counterparty)}

Analyze this case using your 5-step framework and return your assessment as JSON.
""".strip()


# ===== PROMPT ENGINEERING HELPERS =====

def create_chain_of_thought_framework() -> dict:
    """
    Reference structure mirroring the system prompt's 5-step framework.
    Use this to programmatically inspect or regenerate the prompt structure.
    """
    return {
        "step_1": {
            "name": "Data Review",
            "instruction": "Examine customer profile, account details, and all transactions. Note missing or suspicious data points."
        },
        "step_2": {
            "name": "Pattern Recognition",
            "instruction": "Identify unusual volumes, structuring signals ($9k-$10k), rapid fund movement, and income-vs-transaction mismatches."
        },
        "step_3": {
            "name": "Regulatory Mapping",
            "instruction": "Map findings to BSA 31 CFR 1020.320, FinCEN SAR guidelines, OFAC screening, and 12 CFR 21.11."
        },
        "step_4": {
            "name": "Risk Quantification",
            "instruction": "Assign confidence score 0.0-1.0 and risk level (Low/Medium/High/Critical) based on indicator volume and severity."
        },
        "step_5": {
            "name": "Classification Decision",
            "instruction": "Select the single best-fit category from: Structuring, Sanctions, Fraud, Money_Laundering, Other."
        }
    }


def get_classification_categories() -> dict:
    """
    Standard SAR classification categories with regulatory citations and
    common indicators. Use this for prompt construction and documentation.
    """
    return {
        "Structuring": {
            "description": "Transactions deliberately kept below $10,000 to avoid CTR filing",
            "regulation":  "31 CFR 1010.314",
            "indicators":  [
                "multiple cash deposits just under $10k",
                "split transactions on the same day",
                "pattern repeated across multiple branches"
            ]
        },
        "Sanctions": {
            "description": "Transactions involving OFAC-listed entities or prohibited jurisdictions",
            "regulation":  "50 USC 1701 / OFAC SDN list",
            "indicators":  [
                "counterparty name matches SDN list",
                "transaction routed through sanctioned country",
                "unusual international wire destinations"
            ]
        },
        "Fraud": {
            "description": "Deceptive transactions or identity-related suspicious activity",
            "regulation":  "18 USC 1344",
            "indicators":  [
                "account takeover signals",
                "rapid new-account transactions",
                "inconsistent customer information",
                "unusual login locations"
            ]
        },
        "Money_Laundering": {
            "description": "Layering or integration of illicit funds through complex transaction chains",
            "regulation":  "18 USC 1956",
            "indicators":  [
                "round-trip wire transfers",
                "rapid layering across multiple accounts",
                "transaction volume inconsistent with business type",
                "shell company counterparties"
            ]
        },
        "Other": {
            "description": "Suspicious patterns not fitting the four primary categories",
            "regulation":  "31 CFR 1020.320 (general SAR obligation)",
            "indicators":  [
                "unusual customer behavior without clear typology",
                "customer evasiveness about fund sources",
                "unexplained wealth relative to known occupation"
            ]
        }
    }


# ===== TESTING UTILITIES =====

def test_agent_with_sample_case(client=None):
    """
    End-to-end integration test using a synthetic structuring case.
    Pass an existing client or one will be created from OPENAI_API_KEY.
    """
    print("🧪 Testing Risk Analyst Agent")
    print("=" * 50)

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "OPENAI_API_KEY not set"
        base_url = os.getenv("OPENAI_BASE_URL")
        client = openai.OpenAI(api_key=api_key, base_url=base_url or None)

    customer = CustomerData(
        customer_id="CUST_TEST_001",
        name="Jane Doe",
        date_of_birth="1980-06-15",
        ssn_last_4="7890",
        address="456 Oak Ave, Springfield, IL 62701",
        customer_since="2018-03-01",
        risk_rating="High",
        phone="555-987-6543",
        occupation="Retail Worker",
        annual_income=32000
    )

    account = AccountData(
        account_id="CUST_TEST_001_ACC_1",
        customer_id="CUST_TEST_001",
        account_type="Checking",
        opening_date="2018-03-15",
        current_balance=4200.00,
        average_monthly_balance=3100.00,
        status="Active"
    )

    transactions = [
        TransactionData(
            transaction_id=f"TXN_TEST_{i:03d}",
            account_id="CUST_TEST_001_ACC_1",
            transaction_date=f"2024-11-{i + 1:02d}",
            transaction_type="Cash_Deposit",
            amount=9500.00 - (i * 50),
            description="Cash deposit at branch",
            method="Teller",
            counterparty=None,
            location="Springfield Branch"
        )
        for i in range(5)
    ]

    case = CaseData(
        customer=customer,
        accounts=[account],
        transactions=transactions,
        data_sources={
            "customer_source":    "test_synthetic",
            "account_source":     "test_synthetic",
            "transaction_source": "test_synthetic"
        }
    )

    logger = ExplainabilityLogger(log_file="test_audit.jsonl")
    agent  = RiskAnalystAgent(
        openai_client=client,
        explainability_logger=logger,
        model="gpt-4o"
    )

    print(f"  Case ID:      {case.case_id}")
    print(f"  Customer:     {customer.name} | Income: ${customer.annual_income:,}/yr")
    print(f"  Transactions: {len(transactions)} cash deposits | "
          f"Range: ${min(t.amount for t in transactions):,.0f}–"
          f"${max(t.amount for t in transactions):,.0f}")
    print()

    result = agent.analyze_case(case)

    assert result.classification in {"Structuring", "Sanctions", "Fraud", "Money_Laundering", "Other"}
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.risk_level in {"Low", "Medium", "High", "Critical"}
    assert len(result.key_indicators) > 0
    assert len(result.reasoning) > 0

    print("✅ Schema validation passed")
    print()
    print("📊 Analysis Results:")
    print(f"  Classification : {result.classification}")
    print(f"  Risk Level     : {result.risk_level}")
    print(f"  Confidence     : {result.confidence_score:.2f}")
    print(f"  Key Indicators :")
    for indicator in result.key_indicators:
        print(f"    • {indicator}")
    print(f"  Reasoning      : {result.reasoning[:150]}{'...' if len(result.reasoning) > 150 else ''}")
    print()

    if result.classification == "Structuring":
        print("✅ Correctly identified structuring pattern")
    else:
        print(f"⚠️  Expected Structuring, got {result.classification}")

    if result.risk_level in {"High", "Critical"}:
        print("✅ Risk level appropriately elevated")
    else:
        print(f"⚠️  Expected High/Critical risk, got {result.risk_level}")

    print()
    print(f"📋 Audit log: test_audit.jsonl ({len(logger.entries)} entries)")
    print("=" * 50)


def simple_risk_analyst_smoke_test(client):
    """Quick smoke test — pass the already-initialized client."""
    print("🔍 Risk Analyst Smoke Test")
    print("=" * 45)

    # Step 1
    try:
        logger = ExplainabilityLogger(log_file="smoke_test_audit.jsonl")
        agent  = RiskAnalystAgent(
            openai_client=client,
            explainability_logger=logger,
            model="gpt-4o"
        )
        print("✅ Step 1 — Agent initialized")
    except Exception as e:
        print(f"❌ Step 1 FAILED — Could not initialize agent: {e}")
        return

    # Step 2
    try:
        customer = CustomerData(
            customer_id="CUST_SMOKE_001",
            name="Test User",
            date_of_birth="1975-04-20",
            ssn_last_4="1234",
            address="789 Test St, Newark, NJ 07102",
            customer_since="2020-01-01",
            risk_rating="High",
            occupation="Cashier",
            annual_income=28000
        )
        account = AccountData(
            account_id="CUST_SMOKE_001_ACC_1",
            customer_id="CUST_SMOKE_001",
            account_type="Checking",
            opening_date="2020-01-15",
            current_balance=1500.00,
            average_monthly_balance=1200.00,
            status="Active"
        )
        transactions = [
            TransactionData(
                transaction_id=f"TXN_SMOKE_00{i}",
                account_id="CUST_SMOKE_001_ACC_1",
                transaction_date=f"2024-12-{i:02d}",
                transaction_type="Cash_Deposit",
                amount=9800.00 - (i * 100),
                description="Cash deposit at branch",
                method="Teller",
                counterparty=None,
                location="Newark Branch"
            )
            for i in range(1, 4)
        ]
        case = CaseData(
            customer=customer,
            accounts=[account],
            transactions=transactions,
            data_sources={
                "customer_source":    "smoke_test",
                "account_source":     "smoke_test",
                "transaction_source": "smoke_test"
            }
        )
        print("✅ Step 2 — Test case built")
    except Exception as e:
        print(f"❌ Step 2 FAILED — Could not build test case: {e}")
        return

    # Step 3
    try:
        result = agent.analyze_case(case)
        print("✅ Step 3 — analyze_case() returned without error")
    except Exception as e:
        print(f"❌ Step 3 FAILED — analyze_case() raised: {e}")
        return

    # Step 4
    failures = []
    if result.classification not in {"Structuring", "Sanctions", "Fraud", "Money_Laundering", "Other"}:
        failures.append(f"classification '{result.classification}' not in valid set")
    if not (0.0 <= result.confidence_score <= 1.0):
        failures.append(f"confidence_score {result.confidence_score} out of range")
    if result.risk_level not in {"Low", "Medium", "High", "Critical"}:
        failures.append(f"risk_level '{result.risk_level}' not in valid set")
    if not result.reasoning or not result.reasoning.strip():
        failures.append("reasoning is empty")
    if not result.key_indicators:
        failures.append("key_indicators list is empty")

    if failures:
        print("❌ Step 4 FAILED — Structural validation errors:")
        for f in failures:
            print(f"     • {f}")
        return

    print("✅ Step 4 — Output structure valid")

    # Step 5
    print()
    print("📊 Result Summary:")
    print(f"   Classification : {result.classification}")
    print(f"   Risk Level     : {result.risk_level}")
    print(f"   Confidence     : {result.confidence_score:.2f}")
    print(f"   Indicators     : {len(result.key_indicators)} found")
    for indicator in result.key_indicators:
        print(f"     • {indicator}")
    print(f"   Reasoning      : {result.reasoning[:120]}{'...' if len(result.reasoning) > 120 else ''}")
    print()
    print(f"📋 Audit log entries: {len(logger.entries)}")
    print()
    print("✅ SMOKE TEST PASSED")
    print("=" * 45)


if __name__ == "__main__":
    print("🔍 Risk Analyst Agent Module")
    print("Chain-of-Thought reasoning for suspicious activity classification")
    print()

    framework  = create_chain_of_thought_framework()
    categories = get_classification_categories()

    print("📐 Analysis Framework:")
    for step, details in framework.items():
        print(f"  {step}: {details['name']} — {details['instruction'][:60]}...")

    print()
    print("🏷️  Classification Categories:")
    for name, details in categories.items():
        print(f"  {name}: {details['description']}")

    print()
    run_test = input("Run integration test? Requires OPENAI_API_KEY. (y/N): ").strip().lower()
    if run_test == "y":
        test_agent_with_sample_case()
    else:
        print("Skipping integration test.")