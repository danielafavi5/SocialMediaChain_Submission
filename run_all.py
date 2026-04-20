"""
run_all.py
==========
Master Orchestrator. Handles extraction, training, and offline evaluation.
Run from the project root: python run_all.py
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(name, script_rel):
    script_abs = os.path.join(BASE_DIR, script_rel)
    print(f"\n{'='*70}")
    print(f"  RUNNING STAGE: {name}")
    print(f"  Script: {script_abs}")
    print(f"{'='*70}")
    result = subprocess.run(
        [sys.executable, script_abs],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at stage: {name}")
        sys.exit(result.returncode)

def main():
    print("Starting Unified Forensic Sequence Engine — Full Pipeline Evaluation...\n")

    run_step("Generate Manifest", "scripts/generate_manifest.py")
    run_step("Train Unified Sequence Model", "scripts/export_models.py")
    run_step("Offline Reproducibility Evaluation", "scripts/reproduce_results_offline.py")

    print("\n" + "="*70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
