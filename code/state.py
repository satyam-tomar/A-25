from typing import TypedDict, Optional, List

class WorkflowState(TypedDict):
    code_id: str
    code: str
    retrieved_context: Optional[str]
    bug_lines: Optional[List[int]]
    explanation: Optional[str]
    _bugs: Optional[List[str]]
    retry_count: int
    _verification_passed: bool