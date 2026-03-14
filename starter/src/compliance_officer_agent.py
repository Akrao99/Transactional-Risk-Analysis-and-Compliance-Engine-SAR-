# compliance_officer_agent.py
import json
import openai
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
import os

from foundation_sar import (
    ComplianceOfficerOutput,
    ExplainabilityLogger,
    CaseData,
    RiskAnalystOutput
)

load_dotenv()


class ComplianceOfficerAgent:
    """Compliance Officer agent using ReACT prompting framework."""

    def __init__(self, openai_client, explainability_logger: ExplainabilityLogger, model="gpt-4o"):
        self.client = openai_client
        self.logger = explainability_logger
        self.model  = model

        self.system_prompt = """
You are a Senior Compliance Officer with 20+ years of experience in BSA/AML regulatory
reporting. You specialize in drafting Suspicious Activity Reports that meet FinCEN
requirements and withstand regulatory scrutiny.

You follow the ReACT framework for every case — explicit Reasoning before Action:

═══════════════════════════════════════════
REASONING PHASE — Think before you write
═══════════════════════════════════════════

REASON 1 - REVIEW RISK FINDINGS:
- What classification did the Risk Analyst assign?
- What is the confidence score and risk level?
- Which key indicators were flagged?
- Does the evidence support the classification?

REASON 2 - ASSESS REGULATORY REQUIREMENTS:
- Which regulations apply to this classification?
- BSA 31 CFR 1020.320: SAR mandatory filing triggers
- 12 CFR 21.11: National bank SAR obligations
- FinCEN SAR Instructions: Narrative completeness standards
- OFAC: Sanctions screening requirements if applicable

REASON 3 - IDENTIFY REQUIRED NARRATIVE ELEMENTS:
- WHO: Subject name, DOB, address, occupation
- WHAT: Specific suspicious activity and amounts
- WHEN: Date range of suspicious transactions
- WHERE: Account numbers, branch locations
- WHY: Why activity is suspicious relative to profile
- HOW: Transaction methods and patterns used

REASON 4 - PLAN NARRATIVE STRUCTURE:
- Open with the most suspicious activity
- Support with specific transaction details
- Connect to customer profile anomalies
- Close with filing basis and regulatory hook

═══════════════════════════════════════════
ACTION PHASE — Execute with precision
═══════════════════════════════════════════

ACTION 1 - DRAFT NARRATIVE:
- Write ≤120 words (regulatory standard for concise SARs)
- Use past tense, third person, active voice
- Include specific dollar amounts and dates
- Reference account IDs where relevant
- Avoid conclusions about guilt — report facts only

ACTION 2 - CITE REGULATIONS:
- List every applicable regulation by exact citation
- Match citations to the specific classification type
- Always include the base BSA SAR filing requirement

ACTION 3 - VERIFY COMPLETENESS:
- Confirm all 5 W's are addressed (Who/What/When/Where/Why)
- Confirm at least one regulation is cited
- Confirm narrative is factual and not speculative
- Set completeness_check = true only if ALL criteria met

═══════════════════════════════════════════
OUTPUT FORMAT — Strict JSON only
═══════════════════════════════════════════

CRITICAL: Respond with ONLY a valid JSON object. No explanation, no preamble,
no markdown. Just the raw JSON:

{
    "narrative": "Factual SAR narrative ≤120 words in regulatory language",
    "narrative_reasoning": "Your ReACT reasoning summary (max 500 chars)",
    "regulatory_citations": [
        "31 CFR 1020.320 (BSA SAR Filing)",
        "12 CFR 21.11 (National Bank SAR)",
        "FinCEN SAR Instructions"
    ],
    "completeness_check": true
}
"""

    def generate_compliance_narrative(
        self,
        case_data: CaseData,
        risk_analysis: RiskAnalystOutput
    ) -> ComplianceOfficerOutput:
        """Generate regulatory-compliant SAR narrative using ReACT framework."""
        start_time = datetime.now()

        try:
            user_prompt = self._format_prompt(case_data, risk_analysis)

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.2,       # lower than risk analyst — compliance needs consistency
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_prompt}
                ]
            )

            response_content = response.choices[0].message.content

            try:
                json_str = self._extract_json_from_response(response_content)
                parsed   = json.loads(json_str)
                result   = ComplianceOfficerOutput(**parsed)
            except Exception as parse_error:
                raise ValueError(
                    f"Failed to parse Compliance Officer JSON output: {parse_error}"
                )

            # Post-generation validation (word count limit)
            compliance_check = self._validate_narrative_compliance(result.narrative)
            if not compliance_check["within_limit"]:
                raise ValueError(
                    f"Narrative exceeds 120 word limit (got {compliance_check['word_count']} words)"
                )

            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="ComplianceOfficer",
                action="generate_narrative",
                case_id=case_data.case_id,
                input_data={
                    "customer_id":      case_data.customer.customer_id,
                    "classification":   risk_analysis.classification,
                    "risk_level":       risk_analysis.risk_level,
                    "confidence_score": risk_analysis.confidence_score,
                },
                output_data={
                    "narrative_word_count":   compliance_check["word_count"],
                    "within_word_limit":      compliance_check["within_limit"],
                    "completeness_check":     result.completeness_check,
                    "citation_count":         len(result.regulatory_citations),
                },
                reasoning=result.narrative_reasoning,
                execution_time_ms=execution_time_ms,
                success=True,
                error_message=None,
            )

            return result

        except Exception as e:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.log_agent_action(
                agent_type="ComplianceOfficer",
                action="generate_narrative",
                case_id=case_data.case_id,
                input_data={"customer_id": case_data.customer.customer_id},
                output_data={},
                reasoning="Narrative generation failed",
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

    def _format_transactions_for_compliance(self, transactions: List) -> str:
        """Format transaction list for compliance narrative (numbered lines with amount, type, location, method)."""
        lines = []
        for i, txn in enumerate(transactions, 1):
            line = f"{i}. {txn.transaction_date}: ${txn.amount:,.2f} {txn.transaction_type}"
            if getattr(txn, "location", None):
                line += f" at {txn.location}"
            line += f" via {txn.method}"
            lines.append(line)
        return "\n".join(lines)

    def _format_risk_analysis_for_prompt(self, risk_analysis: RiskAnalystOutput) -> str:
        """Format risk analysis results for the compliance prompt."""
        indicators = "\n".join(
            f"  - {indicator}" for indicator in risk_analysis.key_indicators
        )
        return f"""
RISK ANALYST FINDINGS:
  Classification  : {risk_analysis.classification}
  Risk Level      : {risk_analysis.risk_level}
  Confidence Score: {risk_analysis.confidence_score:.2f}
  Reasoning       : {risk_analysis.reasoning}

KEY SUSPICIOUS INDICATORS:
{indicators}
""".strip()

    def _format_prompt(self, case_data: CaseData, risk_analysis: RiskAnalystOutput) -> str:
        """Build the full user prompt combining case data and risk findings."""
        amounts      = [txn.amount for txn in case_data.transactions]
        total_volume = sum(amounts)
        date_range   = (
            f"{min(t.transaction_date for t in case_data.transactions)} to "
            f"{max(t.transaction_date for t in case_data.transactions)}"
            if case_data.transactions else "N/A"
        )

        txn_lines = "\n".join([
            f"  - {txn.transaction_date}: {txn.transaction_type} "
            f"${txn.amount:,.2f} via {txn.method}"
            + (f" | {txn.location}"     if txn.location     else "")
            + (f" | CP: {txn.counterparty}" if txn.counterparty else "")
            for txn in case_data.transactions
        ])

        account_lines = "\n".join([
            f"  - {acc.account_id} ({acc.account_type}) | "
            f"Balance: ${acc.current_balance:,.2f} | Status: {acc.status}"
            for acc in case_data.accounts
        ])

        annual_income = (
            f"${case_data.customer.annual_income:,}"
            if case_data.customer.annual_income else "Not provided"
        )

        risk_section = self._format_risk_analysis_for_prompt(risk_analysis)

        return f"""
CASE ID : {case_data.case_id}
CREATED : {case_data.case_created_at}

=== CUSTOMER PROFILE ===
Name           : {case_data.customer.name}
Customer ID    : {case_data.customer.customer_id}
Date of Birth  : {case_data.customer.date_of_birth}
Address        : {case_data.customer.address}
Occupation     : {case_data.customer.occupation or 'Not provided'}
Annual Income  : {annual_income}
Risk Rating    : {case_data.customer.risk_rating}
Customer Since : {case_data.customer.customer_since}

=== ACCOUNTS ===
{account_lines}

=== TRANSACTIONS ({len(case_data.transactions)} total | {date_range}) ===
{txn_lines}

=== FINANCIAL SUMMARY ===
Total Volume   : ${total_volume:,.2f}
Date Range     : {date_range}
Tx Count       : {len(case_data.transactions)}

=== {risk_section} ===

Using the ReACT framework, generate a regulatory-compliant SAR narrative
for FinCEN submission. Narrative must be ≤120 words. Return JSON only.
""".strip()

    def _validate_narrative_compliance(self, narrative: str) -> Dict[str, Any]:
        """Validate narrative meets regulatory requirements."""
        words     = narrative.split()
        word_count = len(words)

        # Check for required regulatory elements
        narrative_lower = narrative.lower()
        elements_found = {
            "who":   any(w in narrative_lower for w in ["customer", "subject", "account holder"]),
            "what":  any(w in narrative_lower for w in ["deposit", "transfer", "transaction", "withdrew", "wire"]),
            "when":  any(c.isdigit() for c in narrative),   # date digits present
            "where": any(w in narrative_lower for w in ["account", "branch", "bank"]),
            "why":   any(w in narrative_lower for w in ["suspicious", "unusual", "inconsistent", "structuring", "below"]),
        }

        return {
            "word_count":       word_count,
            "within_limit":     word_count <= 120,
            "elements_found":   elements_found,
            "all_elements":     all(elements_found.values()),
            "missing_elements": [k for k, v in elements_found.items() if not v],
        }


# ===== REACT PROMPTING HELPERS =====

def create_react_framework() -> dict:
    """Reference structure for the ReACT compliance framework."""
    return {
        "reasoning_phase": {
            "reason_1": "Review risk analyst findings — classification, confidence, indicators",
            "reason_2": "Assess applicable regulations — BSA, FinCEN, OFAC, 12 CFR 21.11",
            "reason_3": "Identify 5 W's — Who/What/When/Where/Why from case data",
            "reason_4": "Plan narrative structure — lead with activity, support with data",
        },
        "action_phase": {
            "action_1": "Draft narrative ≤120 words in regulatory third-person past tense",
            "action_2": "Cite all applicable regulations by exact CFR/USC reference",
            "action_3": "Verify completeness — all 5 W's present, no speculative language",
        }
    }


def get_regulatory_requirements() -> dict:
    """Key regulatory requirements for SAR narratives."""
    return {
        "word_limit": 120,
        "required_elements": [
            "Customer identification (name, DOB, address)",
            "Suspicious activity description with amounts",
            "Transaction dates and date range",
            "Account identifiers",
            "Explanation of why activity is suspicious"
        ],
        "terminology": [
            "suspicious activity",
            "regulatory threshold",
            "financial institution",
            "Bank Secrecy Act",
            "structuring",
            "currency transaction report"
        ],
        "citations": [
            "31 CFR 1020.320 (BSA SAR Filing)",
            "12 CFR 21.11 (National Bank SAR)",
            "FinCEN SAR Instructions",
            "31 CFR 1010.314 (Structuring)",
            "18 USC 1956 (Money Laundering)",
        ]
    }


def validate_word_count(text: str, max_words: int = 120) -> bool:
    """Utility: check narrative is within word limit."""
    return len(text.split()) <= max_words


# ===== TESTING UTILITIES =====

def test_narrative_generation(client=None):
    """End-to-end test using a synthetic structuring case."""
    from foundation_sar import (
        CustomerData, AccountData, TransactionData, CaseData
    )

    print("🧪 Testing Compliance Officer Agent")
    print("=" * 50)

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "OPENAI_API_KEY not set"
        base_url = os.getenv("OPENAI_BASE_URL")
        client = openai.OpenAI(api_key=api_key, base_url=base_url or None)

    # Build synthetic case
    customer = CustomerData(
        customer_id="CUST_CO_001",
        name="Jane Doe",
        date_of_birth="1980-06-15",
        ssn_last_4="7890",
        address="456 Oak Ave, Springfield, IL 62701",
        customer_since="2018-03-01",
        risk_rating="High",
        occupation="Retail Worker",
        annual_income=32000
    )
    account = AccountData(
        account_id="CUST_CO_001_ACC_1",
        customer_id="CUST_CO_001",
        account_type="Checking",
        opening_date="2018-03-15",
        current_balance=4200.00,
        average_monthly_balance=3100.00,
        status="Active"
    )
    transactions = [
        TransactionData(
            transaction_id=f"TXN_CO_{i:03d}",
            account_id="CUST_CO_001_ACC_1",
            transaction_date=f"2024-11-{i + 1:02d}",
            transaction_type="Cash_Deposit",
            amount=9500.00 - (i * 50),
            description="Cash deposit at branch",
            method="Teller",
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

    # Synthetic risk analyst output
    risk_analysis = RiskAnalystOutput(
        classification="Structuring",
        confidence_score=0.95,
        reasoning="Five cash deposits ranging $9300-$9500 over consecutive days, "
                  "well below $10k CTR threshold. Income of $32k inconsistent with volume.",
        key_indicators=[
            "Multiple cash deposits just below $10,000 CTR threshold",
            "Consecutive daily deposits suggest deliberate structuring",
            "Total volume ($46,500) inconsistent with $32k annual income",
            "Retail worker occupation does not explain cash volume"
        ],
        risk_level="High"
    )

    logger = ExplainabilityLogger(log_file="compliance_test_audit.jsonl")
    agent  = ComplianceOfficerAgent(
        openai_client=client,
        explainability_logger=logger,
        model="gpt-4o"
    )

    result = agent.generate_compliance_narrative(case, risk_analysis)

    # Validate
    word_count = len(result.narrative.split())
    assert isinstance(result, ComplianceOfficerOutput)
    assert word_count <= 120,          f"Narrative exceeds 120 words: {word_count}"
    assert len(result.regulatory_citations) > 0, "No regulatory citations"
    assert result.completeness_check,  "Completeness check failed"
    assert len(result.narrative) > 0,  "Empty narrative"

    print("✅ Schema validation passed")
    print()
    print("📄 Generated SAR Narrative:")
    print("-" * 45)
    print(result.narrative)
    print("-" * 45)
    print(f"   Word count      : {word_count}/120")
    print(f"   Completeness    : {result.completeness_check}")
    print(f"   Citations       : {len(result.regulatory_citations)}")
    for cite in result.regulatory_citations:
        print(f"     • {cite}")
    print(f"   Reasoning       : {result.narrative_reasoning[:120]}...")
    print()
    print(f"📋 Audit log entries: {len(logger.entries)}")
    print("=" * 50)


def simple_compliance_smoke_test(client):
    """Quick smoke test — pass the already-initialized client."""
    from foundation_sar import (
        CustomerData, AccountData, TransactionData, CaseData
    )

    print("🔍 Compliance Officer Smoke Test")
    print("=" * 45)

    # Step 1
    try:
        logger = ExplainabilityLogger(log_file="compliance_smoke_audit.jsonl")
        agent  = ComplianceOfficerAgent(
            openai_client=client,
            explainability_logger=logger,
            model="gpt-4o"
        )
        print("✅ Step 1 — Agent initialized")
    except Exception as e:
        print(f"❌ Step 1 FAILED: {e}")
        return

    # Step 2
    try:
        customer = CustomerData(
            customer_id="CUST_SMOKE_CO_001",
            name="Test Subject",
            date_of_birth="1978-09-12",
            ssn_last_4="5678",
            address="123 Main St, Newark, NJ 07102",
            customer_since="2019-05-01",
            risk_rating="High",
            occupation="Cashier",
            annual_income=30000
        )
        account = AccountData(
            account_id="CUST_SMOKE_CO_001_ACC_1",
            customer_id="CUST_SMOKE_CO_001",
            account_type="Checking",
            opening_date="2019-05-15",
            current_balance=2000.00,
            average_monthly_balance=1800.00,
            status="Active"
        )
        transactions = [
            TransactionData(
                transaction_id=f"TXN_SMOKE_CO_{i}",
                account_id="CUST_SMOKE_CO_001_ACC_1",
                transaction_date=f"2024-12-{i:02d}",
                transaction_type="Cash_Deposit",
                amount=9700.00 - (i * 100),
                description="Cash deposit",
                method="Teller",
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
        risk_analysis = RiskAnalystOutput(
            classification="Structuring",
            confidence_score=0.92,
            reasoning="Three cash deposits just below $10k threshold over three days.",
            key_indicators=[
                "Cash deposits below $10,000 CTR threshold",
                "Consecutive daily pattern",
                "Income inconsistent with deposit volume"
            ],
            risk_level="High"
        )
        print("✅ Step 2 — Test data built")
    except Exception as e:
        print(f"❌ Step 2 FAILED: {e}")
        return

    # Step 3
    try:
        result = agent.generate_compliance_narrative(case, risk_analysis)
        print("✅ Step 3 — generate_compliance_narrative() returned without error")
    except Exception as e:
        print(f"❌ Step 3 FAILED: {e}")
        return

    # Step 4
    failures = []
    word_count = len(result.narrative.split())

    if not result.narrative or not result.narrative.strip():
        failures.append("narrative is empty")
    if word_count > 120:
        failures.append(f"narrative exceeds 120 words ({word_count})")
    if not result.regulatory_citations:
        failures.append("no regulatory citations")
    if not result.narrative_reasoning or not result.narrative_reasoning.strip():
        failures.append("narrative_reasoning is empty")
    if not isinstance(result.completeness_check, bool):
        failures.append("completeness_check is not a boolean")

    if failures:
        print("❌ Step 4 FAILED — Validation errors:")
        for f in failures:
            print(f"     • {f}")
        return

    print("✅ Step 4 — Output structure valid")

    # Step 5
    print()
    print("📊 Result Summary:")
    print(f"   Word Count      : {word_count}/120")
    print(f"   Completeness    : {result.completeness_check}")
    print(f"   Citations       : {len(result.regulatory_citations)}")
    for cite in result.regulatory_citations:
        print(f"     • {cite}")
    print(f"   Narrative       :\n{result.narrative}")
    print()
    print(f"📋 Audit log entries: {len(logger.entries)}")
    print()
    print("✅ SMOKE TEST PASSED")
    print("=" * 45)


if __name__ == "__main__":
    print("✅ Compliance Officer Agent Module")
    print("ReACT prompting for regulatory narrative generation")
    print()

    framework    = create_react_framework()
    requirements = get_regulatory_requirements()

    print("📐 ReACT Framework:")
    for phase, steps in framework.items():
        print(f"  {phase.replace('_', ' ').title()}:")
        for step, desc in steps.items():
            print(f"    {step}: {desc[:70]}...")

    print()
    print(f"📋 Regulatory Requirements:")
    print(f"  Word limit    : {requirements['word_limit']}")
    print(f"  Required      : {len(requirements['required_elements'])} elements")
    print(f"  Citations     : {len(requirements['citations'])} standard refs")

    print()
    run_test = input("Run integration test? Requires OPENAI_API_KEY. (y/N): ").strip().lower()
    if run_test == "y":
        test_narrative_generation()
    else:
        print("Skipping integration test.")