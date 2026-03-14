# Foundation SAR - Core Data Schemas and Utilities
# TODO: Implement core Pydantic schemas and data processing utilities

"""
This module contains the foundational components for SAR processing:

1. Pydantic Data Schemas:
- CustomerData: Customer profile information
- AccountData: Account details and balances
- TransactionData: Individual transaction records
- CaseData: Unified case combining all data sources
- RiskAnalystOutput: Risk analysis results
- ComplianceOfficerOutput: Compliance narrative results

2. Utility Classes:
- ExplainabilityLogger: Audit trail logging
- DataLoader: Combines fragmented data into case objects
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


# =========================
# Customer Schema
# =========================

class CustomerData(BaseModel):
    customer_id: str = Field(..., description="Customer ID")
    name: str = Field(..., description="Full name")
    date_of_birth: str = Field(..., description="Date of birth YYYY-MM-DD")
    ssn_last_4: str = Field(..., description="Last 4 digits of SSN")
    address: str = Field(..., description="Customer address")
    customer_since: str = Field(..., description="Customer since date YYYY-MM-DD")
    risk_rating: Literal["Low", "Medium", "High"] = Field(..., description="Risk rating")

    phone: Optional[str] = Field(None, description="Phone number")
    occupation: Optional[str] = Field(None, description="Customer occupation")
    annual_income: Optional[int] = Field(None, description="Annual income")

    @field_validator("ssn_last_4", mode="before")
    @classmethod
    def coerce_ssn_to_string(cls, v):
        return str(v)

    @field_validator("name", "customer_id")
    @classmethod
    def check_not_empty(cls, v, info):
        if not v or v.strip() == "":
            raise ValueError(f"{info.field_name} cannot be empty")
        return v.strip()

    @field_validator("ssn_last_4")
    @classmethod
    def validate_ssn(cls, v):
        if not v.isdigit() or len(v) != 4:
            raise ValueError("SSN last 4 must be exactly 4 digits")
        return v


# =========================
# Account Schema
# =========================

class AccountData(BaseModel):
    account_id: str = Field(..., description="Unique account identifier")
    customer_id: str = Field(..., description="Customer ID reference")

    account_type: Literal["Checking", "Savings", "Money_Market", "Business_Checking"] = Field(
        ..., description="Type of account"
    )

    opening_date: str = Field(..., description="Account opening date YYYY-MM-DD")

    current_balance: float = Field(..., description="Current balance")
    average_monthly_balance: float = Field(..., description="Average balance")

    status: Literal["Active", "Closed", "Suspended"] = Field(
        ..., description="Account status"
    )

    @field_validator("account_id", "customer_id")
    @classmethod
    def check_id_format(cls, v, info):
        if not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return v.strip()

    @field_validator("opening_date")
    @classmethod
    def validate_date_format(cls, v):
        if len(v) != 10 or v[4] != "-" or v[7] != "-":
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v


# =========================
# Transaction Schema
# =========================

class TransactionData(BaseModel):
    transaction_id: str = Field(..., description="Transaction ID")
    account_id: str = Field(..., description="Account reference")

    transaction_date: str = Field(..., description="Transaction date YYYY-MM-DD")

    transaction_type: str = Field(..., description="Transaction type")
    amount: float = Field(..., description="Transaction amount")

    description: str = Field(..., description="Transaction description")
    method: str = Field(..., description="Transaction method")

    counterparty: Optional[str] = Field(None, description="Counterparty")
    location: Optional[str] = Field(None, description="Transaction location")

    @field_validator("counterparty", "location", mode="before")
    @classmethod
    def coerce_nan_to_none(cls, v):
        import math

        if v is None:
            return None

        if isinstance(v, float) and math.isnan(v):
            return None

        return v


# =========================
# Case Schema
# =========================

class CaseData(BaseModel):
    case_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Case identifier",
    )

    customer: CustomerData = Field(...)

    accounts: List[AccountData] = Field(...)

    transactions: List[TransactionData] = Field(...)

    case_created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Case creation timestamp",
    )

    data_sources: Dict[str, str] = Field(...)

    @field_validator("transactions")
    @classmethod
    def validate_transactions_not_empty(cls, v):
        if not v:
            raise ValueError(
                "Transactions list cannot be empty — at least one suspicious transaction required"
            )
        return v

    @field_validator("accounts")
    @classmethod
    def validate_accounts_same_customer(cls, v):
        if not v:
            raise ValueError("Accounts list cannot be empty")

        customer_ids = {account.customer_id for account in v}

        if len(customer_ids) > 1:
            raise ValueError(
                f"Accounts belong to multiple customers: {customer_ids}"
            )

        return v

    @model_validator(mode="after")
    def validate_transactions_belong_to_accounts(self):

        account_ids = {a.account_id for a in self.accounts}

        invalid_tx = [
            t.transaction_id
            for t in self.transactions
            if t.account_id not in account_ids
        ]

        if invalid_tx:
            raise ValueError(
                f"Transactions reference unknown accounts: {invalid_tx}"
            )

        return self


# =========================
# Risk Analyst Output
# =========================

class RiskAnalystOutput(BaseModel):
    classification: Literal[
        "Structuring",
        "Sanctions",
        "Fraud",
        "Money_Laundering",
        "Other",
    ]

    confidence_score: float = Field(..., ge=0.0, le=1.0)

    reasoning: str = Field(..., max_length=500)

    key_indicators: List[str]

    risk_level: Literal["Low", "Medium", "High", "Critical"]

    @field_validator("key_indicators")
    @classmethod
    def validate_indicators(cls, v):

        if not v:
            raise ValueError("key_indicators cannot be empty")

        if any(i.strip() == "" for i in v):
            raise ValueError("Indicators cannot contain blank strings")

        return v


# =========================
# Compliance Output
# =========================

class ComplianceOfficerOutput(BaseModel):

    narrative: str = Field(..., max_length=1000)

    narrative_reasoning: str = Field(..., max_length=500)

    regulatory_citations: List[str]

    completeness_check: bool

    @field_validator("regulatory_citations")
    @classmethod
    def validate_citations(cls, v):

        if not v:
            raise ValueError("At least one regulation must be cited")

        if any(c.strip() == "" for c in v):
            raise ValueError("Blank citations not allowed")

        return v

    @field_validator("narrative")
    @classmethod
    def validate_narrative(cls, v):

        if not v or v.strip() == "":
            raise ValueError("Narrative cannot be empty")

        return v.strip()

    @model_validator(mode="after")
    def validate_completeness(self):

        if self.completeness_check:

            if len(self.narrative.split()) < 10:
                raise ValueError(
                    "Narrative too short to be considered complete"
                )

            if len(self.regulatory_citations) < 1:
                raise ValueError(
                    "No regulatory citations provided"
                )

        return self


# =========================
# Explainability Logger
# =========================

class ExplainabilityLogger:

    def __init__(self, log_file: str = "sar_audit.jsonl"):
        self.log_file = log_file
        self.entries: List[Dict] = []

    def log_agent_action(
        self,
        agent_type: str,
        action: str,
        case_id: str,
        input_data: Dict,
        output_data: Dict,
        reasoning: str,
        execution_time_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
    ):

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "agent_type": agent_type,
            "action": action,
            "input_summary": str(input_data),
            "output_summary": str(output_data),
            "reasoning": reasoning,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "error_message": error_message,
        }

        self.entries.append(entry)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


# =========================
# Data Loader
# =========================

class DataLoader:

    def __init__(self, explainability_logger: ExplainabilityLogger):
        self.logger = explainability_logger

    def create_case_from_data(
        self,
        customer_data: Dict,
        account_data: List[Dict],
        transaction_data: List[Dict],
    ) -> CaseData:

        start_time = datetime.now()

        try:

            case_id = str(uuid.uuid4())

            customer = CustomerData(**customer_data)

            accounts = [
                AccountData(**acc)
                for acc in account_data
                if acc["customer_id"] == customer.customer_id
            ]

            account_ids = {acc.account_id for acc in accounts}

            transactions = [
                TransactionData(**txn)
                for txn in transaction_data
                if txn["account_id"] in account_ids
            ]

            case = CaseData(
                case_id=case_id,
                customer=customer,
                accounts=accounts,
                transactions=transactions,
                case_created_at=datetime.now(timezone.utc).isoformat(),
                data_sources={
                    "customer_source": "csv_extract",
                    "account_source": "csv_extract",
                    "transaction_source": "csv_extract",
                },
            )

            execution_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case_id,
                input_data={"customer": customer.customer_id},
                output_data={"case_id": case_id},
                reasoning="Case created successfully",
                execution_time_ms=execution_time_ms,
            )

            return case

        except Exception as e:

            execution_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id="UNKNOWN",
                input_data={},
                output_data={},
                reasoning="Case creation failed",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e),
            )

            raise


# =========================
# CSV Loader
# =========================

def load_csv_data(data_dir: str = "data/"):

    try:

        customers_df = pd.read_csv(f"{data_dir}/customers.csv")
        accounts_df = pd.read_csv(f"{data_dir}/accounts.csv")
        transactions_df = pd.read_csv(f"{data_dir}/transactions.csv")

        return customers_df, accounts_df, transactions_df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found: {e}")

    except Exception as e:
        raise Exception(f"Error loading CSV data: {e}")


if __name__ == "__main__":

    print("🏗️ Foundation SAR Module")
    print("Core data schemas and utilities for SAR processing")