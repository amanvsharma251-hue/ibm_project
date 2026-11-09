"""
granite_llm.py
==============
IBM Granite LLM integration via IBM watsonx.ai (IBM Cloud Lite tier).

Uses the official `langchain-ibm` package which wraps the ibm-watsonx-ai SDK.
The WatsonxLLM class is fully LangChain-compatible and works with chains,
LCEL pipelines, and direct calls.

Setup:
  1. Create a free IBM Cloud account: https://cloud.ibm.com/registration
  2. Provision Watson Machine Learning (Lite plan — free)
  3. Create a watsonx.ai project and note the Project ID
  4. Generate an API key at https://cloud.ibm.com/iam/apikeys
  5. Fill in config.env with your credentials
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv("config.env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IBM watsonx.ai / Granite configuration
# ---------------------------------------------------------------------------
WATSONX_API_KEY      = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID   = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL          = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
GRANITE_MODEL_ID     = os.getenv("GRANITE_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

GRANITE_MAX_TOKENS   = int(os.getenv("GRANITE_MAX_NEW_TOKENS", "1024"))
GRANITE_TEMPERATURE  = float(os.getenv("GRANITE_TEMPERATURE", "0.1"))
GRANITE_TOP_P        = float(os.getenv("GRANITE_TOP_P", "0.9"))
GRANITE_TOP_K        = int(os.getenv("GRANITE_TOP_K", "50"))
GRANITE_REP_PENALTY  = float(os.getenv("GRANITE_REPETITION_PENALTY", "1.1"))


# ---------------------------------------------------------------------------
# LangChain-compatible Granite LLM
# ---------------------------------------------------------------------------

def get_granite_llm():
    """
    Returns a LangChain-compatible WatsonxLLM backed by IBM Granite on watsonx.ai.

    The returned object can be used directly in LCEL chains:
        response = llm.invoke("Your prompt here")

    Raises:
        EnvironmentError: If required environment variables are missing.
        ImportError:      If langchain-ibm is not installed.
    """
    if not WATSONX_API_KEY or WATSONX_API_KEY == "your_ibm_cloud_api_key_here":
        raise EnvironmentError(
            "WATSONX_API_KEY is not set.\n"
            "1. Get your API key from https://cloud.ibm.com/iam/apikeys\n"
            "2. Set it in config.env:  WATSONX_API_KEY=your_actual_key"
        )
    if not WATSONX_PROJECT_ID or WATSONX_PROJECT_ID == "your_watsonx_project_id_here":
        raise EnvironmentError(
            "WATSONX_PROJECT_ID is not set.\n"
            "Find it in your watsonx.ai project → Settings tab.\n"
            "Set it in config.env:  WATSONX_PROJECT_ID=your_project_id"
        )

    try:
        from langchain_ibm import WatsonxLLM
    except ImportError:
        raise ImportError(
            "langchain-ibm is not installed. Run:\n"
            "  pip install langchain-ibm"
        )

    logger.info("Initialising IBM Granite LLM: model=%s, url=%s", GRANITE_MODEL_ID, WATSONX_URL)

    llm = WatsonxLLM(
        model_id=GRANITE_MODEL_ID,
        url=WATSONX_URL,
        apikey=WATSONX_API_KEY,
        project_id=WATSONX_PROJECT_ID,
        params={
            "max_new_tokens": GRANITE_MAX_TOKENS,
            "temperature":    GRANITE_TEMPERATURE,
            "top_p":          GRANITE_TOP_P,
            "top_k":          GRANITE_TOP_K,
            "repetition_penalty": GRANITE_REP_PENALTY,
            "decoding_method": "greedy" if GRANITE_TEMPERATURE == 0 else "sample",
        },
    )

    logger.info("IBM Granite LLM ready.")
    return llm


# ---------------------------------------------------------------------------
# System Prompt — College Admission Assistant persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an intelligent and helpful College Admission Assistant.
Your role is to assist prospective students with accurate, friendly, and concise answers about:
  - Admission policies and eligibility criteria
  - Available courses and programs
  - Fee structure and scholarships
  - Important dates and deadlines
  - Application process and required documents
  - Hostel, campus life, and placements

Guidelines:
1. Base your answers ONLY on the provided context from the knowledge base.
2. If the answer is not in the context, say "I don't have that specific information — please contact the admissions office directly."
3. Be concise, factual, and friendly. Use bullet points when listing multiple items.
4. Never invent fees, dates, or policies.
5. Do NOT mention any specific college name — refer to it as "the institution" or "this college".
6. If a question is ambiguous, ask a clarifying question.
7. Always end with an offer to help further.

You are powered by IBM Granite, an enterprise-grade AI model from IBM watsonx.ai.
"""


def build_rag_prompt(user_question: str, context: str) -> str:
    """
    Build the full prompt string for a RAG query.

    Args:
        user_question: The student's natural-language question.
        context:       Formatted retrieval context from the vector store.

    Returns:
        A formatted prompt string ready to be sent to the Granite LLM.
    """
    prompt = f"""{SYSTEM_PROMPT}

=== RETRIEVED KNOWLEDGE BASE CONTEXT ===
{context}
=== END OF CONTEXT ===

Student Question: {user_question}

Assistant Answer:"""
    return prompt


# ---------------------------------------------------------------------------
# Quick smoke-test (run with: python granite_llm.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing IBM Granite LLM connection …\n")
    try:
        llm = get_granite_llm()
        test_prompt = (
            "You are a helpful assistant. "
            "In one sentence, what is IBM watsonx.ai?"
        )
        response = llm.invoke(test_prompt)
        print("Response from Granite:\n", response)
    except EnvironmentError as e:
        print(f"[Config Error] {e}")
    except Exception as e:
        print(f"[Error] {e}")
