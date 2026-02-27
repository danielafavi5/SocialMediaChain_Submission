# TrueFake-IJCNN25 — Forensic Platform Identification

Reproduction package for the study:  
*Multi-Clue Reconstruction of Sharing Chains for Social Media Images*  
University of Trento — Multimedia Data Security 2026

---

## How to Run

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Analyze a single image**

```bash
python analyze_image.py --image samples/D01_I_nat_0001.chain_502c0eb0_1771466632.step3.telegram.jpg
```

**3. Run the full offline reproducibility check**

```bash
python scripts/reproduce_results_offline.py
```

This script loads the pre-trained models from `/models`, runs predictions on all images in `/samples`, and prints a mini-report confirming the accuracy figures.

---

## How to Grade Offline

The package is fully self-contained. No internet connection or re-training is required.

| Component | Location |
|---|---|
| Pre-trained C1 model (surface RF) | `models/c1_surface.joblib` |
| Pre-trained C2 model (BKS residual RF) | `models/c2_residual.joblib` |
| Adversarial pruning index | `models/pruned_indices.npy` |
| 1,000 ground-truth 2026 test images | `samples/` |
| Ground-truth manifest schema | `manifest_example.json` |
| Reproducibility script | `scripts/reproduce_results_offline.py` |

---

## Training Data Source

To ensure the integrity of our ground-truth labels and prevent historical compression from contaminating the training set, all models were trained on 3,000 high-resolution, uncompressed source images pulled from two recognized forensic benchmarks:
- **RAISE (Research All-around Image Dataset)**
- **VISION Dataset**

These pristine files were transmitted through live platform APIs using the `chained_uploader.py` orchestrator to generate authentic 2024 and 2026 compression artifacts.

---

## Results Summary

| Task | Result |
|---|---|
| C1 surface platform identification (2026) | **89.9% macro-F1** |
| C2 BKS residual classification (mixed) | **95.4% macro-F1** |
| 3-step chain reconstruction | **0.0%** (Forensic Horizon) |

The 0.0% chain accuracy is a mathematical boundary, not a model failure.  
See `research_summary.md` for the full derivation.

---

## Project Structure

```
├── core/                    Feature extractor + BKS fusion logic
├── training/                Training pipeline source
├── audits/                  Chain evaluation scripts
├── models/                  Pre-trained .joblib artifacts
├── assets/                  Diagnostic graphs
├── samples/                 Ground-truth 2026 test images
├── scripts/                 Utility scripts
├── analyze_image.py         Single-image CLI tool
├── export_models.py         Re-train and save models from scratch
├── manifest_example.json    Sample ground-truth manifest (5 chains)
├── methodology.md           Data pipeline and feature engineering details
├── research_summary.md      Forensic Horizon mathematical explanation
├── defense_notes.md         Oral defense Q&A guide
└── requirements.txt
```
