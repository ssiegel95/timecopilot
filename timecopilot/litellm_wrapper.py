"""Create a litellm-compatible model.

This will be a no-op if `pydantic_ai_litellm` is not installed.
"""

import logging
import os

logger = logging.getLogger(__name__)

try:
    from pydantic_ai_litellm import LiteLLMModel
    NO_OP = False
except ImportError:
    logger.warning(
        "pydantic_ai_litellm is not installed — create_litellm_model() will be a no-op."
    )
    NO_OP = True


def create_litellm_model(model: str):
    """Create a LiteLLMModel unless `pydantic_ai_litellm` is unavailable.

    Environment variables used:

    - LITELLM_API_KEY: API key for LiteLLM
    - LITELLM_API_BASE_URL: optional base URL (e.g. https://your.custom.endpoint)
    - LITELLM_CUSTOM_LLM_PROVIDER: optional provider name (e.g. "litellm_proxy")

    Args:
        model (str): Model name.

    Returns:
        LiteLLMModel | str: A configured LiteLLMModel, or the model string if NO_OP.
    """
    if NO_OP:
        # Return plain model string when dependency is missing
        return model

    api_key = os.getenv("LITELLM_API_KEY", "your_api_key")
    api_base = os.getenv("LITELLM_API_BASE_URL")
    custom_provider = os.getenv("LITELLM_CUSTOM_LLM_PROVIDER")

    kwargs = {"model_name": model, "api_key": api_key}

    if api_base:
        kwargs["api_base"] = api_base
    if custom_provider:
        kwargs["custom_llm_provider"] = custom_provider

    return LiteLLMModel(**kwargs)
