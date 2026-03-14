# SAR Processing System - Source Code Package
"""
Suspicious Activity Report (SAR) Processing System

This package contains the core modules for the AI-powered
SAR processing system for financial crime detection.
"""

__version__ = "1.0.0"

import os
from typing import Optional

def create_openai_client():
    """
    Create an OpenAI client using environment variables.

    Uses OPENAI_API_KEY and optional OPENAI_BASE_URL (for custom endpoints).

    Returns:
        openai.OpenAI: Configured OpenAI client instance

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
        ImportError: If openai package is not installed
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai package is required. Install with: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Add it to your .env file (see .env.template)."
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, base_url=base_url or None)
    return client

# Backward compatibility
create_vocareum_openai_client = create_openai_client
