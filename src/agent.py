"""
agent.py
========
College Admission Agent — orchestrates the RAG pipeline and IBM Granite LLM.

Flow:
  1. User asks a question
  2. RAG engine retrieves relevant chunks from the knowledge base
  3. Chunks are formatted into a prompt with the Granite system persona
  4. IBM Granite generates a grounded, context-aware answer
  5. The answer (+ sources) is returned to the caller
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Allow running from either project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv("config.env")

from rag_engine import get_or_build_vector_store, retrieve_relevant_chunks, format_context
from granite_llm import get_granite_llm, build_rag_prompt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Singleton instances (initialised once, reused across queries)
# ---------------------------------------------------------------------------
_vector_store = None
_llm = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        logger.info("Initialising vector store …")
        _vector_store = get_or_build_vector_store()
    return _vector_store


def _get_llm():
    global _llm
    if _llm is None:
        logger.info("Initialising IBM Granite LLM …")
        _llm = get_granite_llm()
    return _llm


# ---------------------------------------------------------------------------
# Core answer function
# ---------------------------------------------------------------------------

def answer_query(
    question: str,
    top_k: Optional[int] = None,
    return_sources: bool = True,
) -> dict:
    """
    Answer a college admission question using RAG + IBM Granite.

    Args:
        question:       The student's natural-language question.
        top_k:          Override number of retrieved chunks (default from env).
        return_sources: Whether to include source file names in the response.

    Returns:
        {
            "answer":  str,            # Granite's generated answer
            "sources": list[str],      # Source file names (if return_sources=True)
            "context": str,            # Raw context passed to the LLM
            "error":   str | None,     # Error message if something went wrong
        }
    """
    if not question or not question.strip():
        return {
            "answer": "Please ask me a question about college admissions!",
            "sources": [],
            "context": "",
            "error": None,
        }

    try:
        # Step 1: Retrieve relevant chunks
        vector_store = _get_vector_store()
        rag_top_k = top_k or int(os.getenv("RAG_TOP_K", "5"))
        relevant_docs = retrieve_relevant_chunks(question, vector_store, top_k=rag_top_k)

        # Step 2: Format context
        context = format_context(relevant_docs)

        # Step 3: Build prompt
        prompt = build_rag_prompt(question, context)
        logger.debug("Prompt length: %d characters", len(prompt))

        # Step 4: Generate answer with Granite
        llm = _get_llm()
        raw_answer = llm.invoke(prompt)

        # Step 5: Clean up the response
        answer = raw_answer.strip() if raw_answer else "I'm sorry, I couldn't generate an answer."

        # Collect source names
        sources = []
        if return_sources:
            seen = set()
            for doc in relevant_docs:
                src = doc.metadata.get("source", "")
                name = Path(src).name if src else "knowledge base"
                if name not in seen:
                    sources.append(name)
                    seen.add(name)

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "error": None,
        }

    except EnvironmentError as e:
        # Missing IBM credentials — surface clearly to the user
        error_msg = str(e)
        logger.error("Configuration error: %s", error_msg)
        return {
            "answer": (
                "⚠️ **Configuration Required**\n\n"
                "IBM watsonx.ai credentials are not configured.\n\n"
                f"{error_msg}\n\n"
                "Please update `config.env` with your IBM Cloud credentials and restart."
            ),
            "sources": [],
            "context": "",
            "error": error_msg,
        }

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}"
        full_trace = traceback.format_exc()
        logger.exception("Error answering query: %s", question)
        return {
            "answer": (
                f"⚠️ **Error:** `{error_msg}`\n\n"
                "Check the terminal for the full traceback.\n\n"
                "Common causes:\n"
                "- Wrong API key or Project ID in `config.env`\n"
                "- IBM watsonx.ai service not provisioned on your account\n"
                "- Network / firewall blocking the request"
            ),
            "sources": [],
            "context": "",
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# Conversation memory helper (simple turn-based history)
# ---------------------------------------------------------------------------

class AdmissionChatSession:
    """
    Lightweight in-memory chat session with turn history.
    Injects recent history into the question for context continuity.
    """

    MAX_HISTORY_TURNS = 5   # keep last N Q&A pairs in context

    def __init__(self):
        self.history: list[dict] = []   # [{role, content}, ...]

    def chat(self, user_message: str) -> dict:
        """
        Process a user message, inject history, return answer dict.
        """
        # Build context-aware question if we have history
        enriched_question = self._enrich_with_history(user_message)

        result = answer_query(enriched_question)

        # Store turn in history
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant",  "content": result["answer"]})

        # Trim history to MAX_HISTORY_TURNS
        max_msgs = self.MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

        return result

    def _enrich_with_history(self, question: str) -> str:
        """
        Prepend recent conversation turns to the question so the LLM has context
        for follow-up questions (e.g., "What about the fees for that program?").
        """
        if not self.history:
            return question
        recent = self.history[-(self.MAX_HISTORY_TURNS * 2):]
        history_text = "\n".join(
            f"{'Student' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )
        return (
            f"[Previous conversation]\n{history_text}\n\n"
            f"[New question]\n{question}"
        )

    def reset(self):
        """Clear conversation history."""
        self.history = []


# ---------------------------------------------------------------------------
# CLI interactive mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  College Admission Agent (IBM Granite RAG)")
    print("  Springfield Institute of Technology")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    session = AdmissionChatSession()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print("Thank you for using the SIT Admission Assistant. Good luck!")
            break

        if not user_input:
            continue

        print("\nAssistant: ", end="", flush=True)
        result = session.chat(user_input)
        print(result["answer"])

        if result["sources"]:
            print(f"\n📚 Sources: {', '.join(result['sources'])}")

        if result["error"]:
            print(f"\n[Debug] Error: {result['error']}")
