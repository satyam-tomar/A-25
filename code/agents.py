import json
import re
from langchain_openai import ChatOpenAI
from state import WorkflowState
from logger import get_logger
import mcp_client

from dotenv import load_dotenv
load_dotenv()

log = get_logger("agents")

llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

llm_text = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.0,
)

def _invoke_llm(prompt: str) -> str:
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content.strip()

def _invoke_llm_text(prompt: str) -> str:
    response = llm_text.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content.strip()

def _parse_json_response(content: str, fallback_bug_lines: list, fallback_bugs: list):
    try:
        parsed = json.loads(content)
        return parsed.get("bug_lines", fallback_bug_lines), parsed.get("bugs", fallback_bugs)
    except Exception:
        return fallback_bug_lines, fallback_bugs

def _extract_tokens(code: str) -> str:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
    unique_tokens = list(dict.fromkeys(tokens))
    return " ".join(unique_tokens[:20])

def _generate_semantic_queries(code: str) -> list[str]:
    """Use LLM to generate targeted retrieval queries specific to this code."""
    prompt = f"""You are a documentation retrieval assistant for the SmartRDI hardware test API.

Given this code snippet, generate 4 specific search queries that would retrieve the most relevant documentation.
Each query must target a specific method or pattern actually present in the code.
Focus on: correct parameter values, argument order, valid modes/flags, placement rules, valid unit types.
Do NOT generate generic queries. Each query must name the specific method or variable from the code.

Code:
{code}

Return exactly 4 queries, one per line, no numbering, no explanation."""

    result = _invoke_llm_text(prompt)
    queries = [q.strip() for q in result.strip().splitlines() if q.strip()]
    return queries[:6]

def retrieval_agent(state: WorkflowState) -> dict:
    log.info(f"[RETRIEVAL AGENT] Starting for code_id={state['code_id']}")
    code = state["code"]

    # Query 1: token-based (original approach)
    token_query = _extract_tokens(code)
    log.debug(f"[RETRIEVAL AGENT] Token query: {token_query}")

    # Queries 2-4: LLM-generated semantic queries
    semantic_queries = _generate_semantic_queries(code)
    log.debug(f"[RETRIEVAL AGENT] Semantic queries: {semantic_queries}")

    all_queries = [token_query] + semantic_queries

    # Deduplicate context chunks across all queries
    seen_texts = set()
    context_parts = []

    for query in all_queries:
        results = mcp_client.search_documents(query)
        for item in results[:10]:
            if isinstance(item, dict) and "text" in item:
                text = item["text"]
                # Deduplicate by first 80 chars
                key = text[:100]
                if key not in seen_texts:
                    seen_texts.add(key)
                    context_parts.append(text)
        if len(context_parts) >= 60:
            break

    retrieved_context = "\n---\n".join(context_parts[:60])
    log.info(f"[RETRIEVAL AGENT] Collected {len(context_parts)} unique context chunk(s) from {len(all_queries)} queries")
    return {"retrieved_context": retrieved_context}


def bug_finder_agent(state: WorkflowState) -> dict:
    log.info(f"[BUG FINDER AGENT] Starting for code_id={state['code_id']} (retry_count={state.get('retry_count', 0)})")
    code = state["code"]
    context = state.get("retrieved_context", "")
    numbered_code = "\n".join(f"{i+1}: {line}" for i, line in enumerate(code.splitlines()))

    prompt = f"""You are a strict SmartRDI code bug detector.

Documentation context (your ONLY source of truth):
{context if context else "No context available."}

Code (line numbers shown):
{numbered_code}

RULES:
- Only flag a line if the documentation explicitly shows it is wrong
- Check for: wrong method names, wrong argument order, wrong units, wrong parameter values,
  wrong mode flags, calls in wrong lifecycle position (inside/outside RDI_BEGIN/END),
  inverted argument polarity, exceeded value ranges
- Empty lines and comments are NEVER bugs
- When uncertain, skip it
Do NOT report generic syntax errors like:
- missing semicolon
- spacing issues
- formatting problems

Focus on:
- API misuse
- lifecycle errors (RDI_BEGIN / RDI_END)
- parameter order issues
- invalid SmartRDI function usage
- measurement configuration errors
- Each bug: 3-6 words max (e.g. "wrong unit uA not V", "inverted iClamp args", "vecEditMode must be VTT")

-you should focus on the retrieved documents to find the bug instead of auto finding bugs

Return ONLY JSON:
{{
  "bug_lines": [2, 5],
  "bugs": ["wrong unit uA not V", "inverted lifecycle order"]
}}


bug_lines and bugs must have equal length.
If no bugs: {{"bug_lines": [], "bugs": []}}"""

    log.debug("[BUG FINDER AGENT] Invoking LLM")
    content = _invoke_llm(prompt)
    bug_lines, bugs = _parse_json_response(content, [], [])
    min_len = min(len(bug_lines), len(bugs))
    bug_lines, bugs = bug_lines[:min_len], bugs[:min_len]
    explanation = " | ".join(f"L{bl}: {desc}" for bl, desc in zip(bug_lines, bugs))
    log.info(f"[BUG FINDER AGENT] Detected bug_lines={bug_lines} | explanation={explanation}")
    return {"bug_lines": bug_lines, "explanation": explanation, "_bugs": bugs}


def reasoning_agent(state: WorkflowState) -> dict:
    log.info(f"[REASONING AGENT] Starting for code_id={state['code_id']}")
    code = state["code"]
    bug_lines = state.get("bug_lines", [])
    bugs = state.get("_bugs", [])
    context = state.get("retrieved_context", "")
    numbered_code = "\n".join(f"{i+1}: {line}" for i, line in enumerate(code.splitlines()))

    if not context:
        log.info("[REASONING AGENT] No context, re-querying MCP")
        extra_results = mcp_client.search_documents(f"constraints rules {code[:100]}")
        if extra_results:
            context = "\n".join(
                item["text"] for item in extra_results[:15]
                if isinstance(item, dict) and "text" in item
            ).strip()
            log.info("[REASONING AGENT] Context enriched")

    prompt = f"""You are a senior SmartRDI code reviewer. Validate the reported bugs strictly.

Documentation context:
{context}

Code:
{numbered_code}

Reported bugs:
{json.dumps({"bug_lines": bug_lines, "bugs": bugs})}

RULES:
- Keep only bugs clearly supported by the documentation context above
- Remove false positives
- Do NOT add new bugs
- Keep descriptions 3-6 words
- bug_lines and bugs must stay equal length

Return ONLY JSON:
{{
  "bug_lines": {json.dumps(bug_lines)},
  "bugs": {json.dumps(bugs)}
}}"""

    log.debug("[REASONING AGENT] Invoking LLM for refinement")
    content = _invoke_llm(prompt)
    bug_lines, bugs = _parse_json_response(content, bug_lines, bugs)
    min_len = min(len(bug_lines), len(bugs))
    bug_lines, bugs = bug_lines[:min_len], bugs[:min_len]
    explanation = " | ".join(f"L{bl}: {desc}" for bl, desc in zip(bug_lines, bugs))
    log.info(f"[REASONING AGENT] Refined | bug_lines={bug_lines} | explanation={explanation}")
    return {"bug_lines": bug_lines, "explanation": explanation, "_bugs": bugs, "retrieved_context": context}


def verification_agent(state: WorkflowState) -> dict:
    log.info(f"[VERIFICATION AGENT] Starting for code_id={state['code_id']}")
    bug_lines = state.get("bug_lines")
    bugs = state.get("_bugs", [])
    retry_count = state.get("retry_count", 0)

    valid = True
    if not isinstance(bug_lines, list):
        log.warning("[VERIFICATION AGENT] FAIL — bug_lines not a list")
        valid = False
    elif len(bug_lines) != len(bugs):
        log.warning("[VERIFICATION AGENT] FAIL — bug_lines/bugs length mismatch")
        valid = False

    if valid:
        log.info(f"[VERIFICATION AGENT] PASSED | bugs={list(zip(bug_lines, bugs))}")
        return {"_verification_passed": True, "retry_count": retry_count}
    else:
        new_retry = retry_count + 1
        log.warning(f"[VERIFICATION AGENT] FAILED — retrying (attempt {new_retry}/3)")
        return {"_verification_passed": False, "retry_count": new_retry}


def should_retry(state: WorkflowState) -> str:
    passed = state.get("_verification_passed", False)
    retry_count = state.get("retry_count", 0)
    if not passed and retry_count < 3:
        log.info(f"[ROUTER] Retrying bug_finder (attempt {retry_count}/3)")
        return "retry"
    log.info("[ROUTER] Routing to output writer")
    return "done"