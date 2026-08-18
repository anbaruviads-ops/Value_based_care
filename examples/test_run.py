"""
Interactive & Batch Test Runner for the ACO AI Assistant Module.
Run this script to test all sample queries and print rich formatted outputs in terminal.
"""

import sys
import json
import time
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ai_assistant import ask
from tests.sample_payloads import SAMPLE_PAYLOADS

def run_all_samples():
    print("=" * 80)
    print(" [ACO ANALYTICS AI ASSISTANT - BATCH TEST RUNNER]")
    print("=" * 80)

    for idx, item in enumerate(SAMPLE_PAYLOADS, 1):
        print(f"\n[{idx}/{len(SAMPLE_PAYLOADS)}] Testing: {item['name']}")
        print(f" -> Question: {item['payload']['question']}")
        print(f" -> Context: {json.dumps(item['payload'].get('context', {}))}")

        start_time = time.time()
        result = ask(
            question=item["payload"]["question"],
            context=item["payload"].get("context")
        )
        elapsed = time.time() - start_time

        print(f" -> Response Time: {elapsed:.2f}s")
        print(f" -> Detected Intent: {result.get('intent')}")
        print(f" -> Sources Used: {result.get('sources')}")
        print(f" -> AI Answer:\n{result.get('answer')}")
        if result.get("warnings"):
            print(f" -> Warnings: {result.get('warnings')}")
        print("-" * 80)

def run_interactive_mode():
    print("\n" + "=" * 80)
    print(" [INTERACTIVE CHATBOT MODE] (Type 'exit' to quit)")
    print("=" * 80)

    session_id = "interactive_session"
    current_context = {"page": "ACO Explorer"}

    while True:
        try:
            user_input = input("\n>> Enter question (or 'exit'): ").strip()
            if not user_input or user_input.lower() in ("exit", "quit", "q"):
                break

            result = ask(
                question=user_input,
                context=current_context,
                session_id=session_id
            )

            print(f"\n[Intent: {result.get('intent')} | Sources: {result.get('sources')}]")
            print(f"\n{result.get('answer')}\n")
            if result.get("warnings"):
                print(f"Warnings: {result.get('warnings')}")

        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_mode()
    else:
        run_all_samples()
