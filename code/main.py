import sys
import os
import pandas as pd
from workflow import build_workflow
from state import WorkflowState
from logger import get_logger

log = get_logger("main")

def run(input_csv: str):
    log.info(f"Loading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    log.info(f"Loaded {len(df)} code sample(s)")

    workflow = build_workflow()
    log.info("Workflow graph compiled successfully")

    results = []

    for _, row in df.iterrows():
        code_id = str(row["code_id"])
        code = str(row["code"])

        log.info(f"{'='*60}")
        log.info(f"Processing code_id={code_id}")
        log.debug(f"Code snippet: {code[:80]}{'...' if len(code) > 80 else ''}")

        initial_state: WorkflowState = {
            "code_id": code_id,
            "code": code,
            "retrieved_context": None,
            "bug_lines": None,
            "explanation": None,
            "_bugs": None,
            "retry_count": 0,
            "_verification_passed": False,
        }

        try:
            final_state = workflow.invoke(initial_state)
            bug_lines = final_state.get("bug_lines") or []
            bugs      = final_state.get("_bugs") or []

            # Align lengths just in case
            min_len   = min(len(bug_lines), len(bugs))
            bug_lines = bug_lines[:min_len]
            bugs      = bugs[:min_len]

            log.info(f"code_id={code_id} completed | bug_lines={bug_lines} | bugs={bugs}")
        except Exception as e:
            log.error(f"code_id={code_id} workflow failed: {e}")
            bug_lines = []
            bugs = []

        results.append({
            "code_id":    code_id,
            "bug_lines":  ", ".join(str(b) for b in bug_lines) if bug_lines else "",
            "bugs":       " | ".join(f"L{bl}: {desc}" for bl, desc in zip(bug_lines, bugs)) if bug_lines else "No bug found",
        })

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output.csv")
    out_df = pd.DataFrame(results, columns=["code_id", "bug_lines", "bugs"])
    out_df.to_csv(output_path, index=False)
    log.info(f"{'='*60}")
    log.info(f"output.csv written to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.error("No input CSV provided. Usage: python main.py <input_csv_path>")
        sys.exit(1)
    run(sys.argv[1])