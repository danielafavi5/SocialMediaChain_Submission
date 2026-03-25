import os
import subprocess
import sys

def run_step(name, cmd):
    print(f"\n{'='*70}")
    print(f"  RUNNING STAGE: {name}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    print(result.stdout)
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at stage: {name}")
        sys.exit(result.returncode)

def main():
    print("Starting End-to-End Pipeline Evaluation...\n")
    
    run_step("Generate Manifest", [sys.executable, "scripts/generate_manifest.py"])
    run_step("Export Models and Split Data", [sys.executable, "export_models.py"])
    run_step("Offline Reproducibility Evaluation", [sys.executable, "scripts/reproduce_results_offline.py"])
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()
