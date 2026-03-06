from langgraph.graph import StateGraph, END
from state import WorkflowState
from logger import get_logger
from agents import (
    retrieval_agent,
    bug_finder_agent,
    reasoning_agent,
    verification_agent,
    should_retry,
)

log = get_logger("workflow")

def build_workflow():
    log.info("Building LangGraph workflow")
    graph = StateGraph(WorkflowState)

    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("bug_finder_agent", bug_finder_agent)
    graph.add_node("reasoning_agent", reasoning_agent)
    graph.add_node("verification_agent", verification_agent)

    graph.set_entry_point("retrieval_agent")
    graph.add_edge("retrieval_agent", "bug_finder_agent")
    graph.add_edge("bug_finder_agent", "reasoning_agent")
    graph.add_edge("reasoning_agent", "verification_agent")

    graph.add_conditional_edges(
        "verification_agent",
        should_retry,
        {
            "retry": "bug_finder_agent",
            "done": END,
        },
    )

    log.info("Workflow graph ready: retrieval → bug_finder → reasoning → verification → (retry | END)")
    return graph.compile()