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

**2. Run the full pipeline from scratch (recommended)**

```bash
python run_all.py
```

This single command generates the manifest, retrains the Unified Sequence Engine, and runs the offline evaluation — fully reproducible from any directory.

**3. Analyze a single image**

```bash
python analyze_image.py --image samples/D01_I_nat_0001.chain_502c0eb0_1771466632.step3.telegram.jpg
```

**4. Run offline evaluation only (requires pre-trained model)**

```bash
python scripts/reproduce_results_offline.py
```

---

## How to Grade Offline

The package is fully self-contained. No internet connection is required for evaluation; `run_all.py` will retrain from the included `samples/` if needed.

| Component | Location |
|---|---|
| Unified Sequence Model (V2) | `models/seq_model.joblib` |
| Surface Classifier Model | `models/surface_model.joblib` |
| 1,000 ground-truth 2026 test images | `samples/` |
| Ground-truth manifest schema | `manifest.json` |
| Reproducibility script | `scripts/reproduce_results_offline.py` |
| Master orchestrator | `run_all.py` |

---

## Training Data Source

To ensure the integrity of our ground-truth labels and prevent historical compression from contaminating the training set, all models were trained on 3,000 high-resolution, uncompressed source images pulled from two recognized forensic benchmarks:
- **RAISE (Research All-around Image Dataset)**
- **VISION Dataset**

These pristine files were transmitted through live platform APIs using the `chained_uploader.py` orchestrator to generate authentic 2024 and 2026 compression artifacts.

---

## Results Summary (V2 Unified Engine)

| Task | Result |
|---|---|
| Step 1 Platform Accuracy | **97.0%** |
| Step 2 Platform Accuracy | **79.1%** |
| Step 3 Platform Accuracy | **85.1%** |
| Exact 3-step Chain Reconstruction | **74.6%** |

### How Benford's Law Distribution Analysis Partially Recovers Discord Traces

Previous versions of this pipeline suffered from a **"Discord Tracing Limit"**: Discord's aggressive low-coefficient quantization table mathematically erased the divisibility ratios that would identify prior platforms, collapsing the chain trace to ~4.5% accuracy.

The V2 architecture partially overcomes this via **Benford's Law analysis** of AC DCT coefficients. When an image's compression history passes through Discord, its violent re-quantization introduces abnormal deviations in the expected leading-digit distribution of AC coefficients (where natural signals follow Benford's Law, with digit 1 appearing ~30% of the time). These distribution anomalies — combined with JPEG container-level marker order fingerprints (`has_dht`, `dqt_before_dht`) — provide a non-divisibility signal that the Random Forest can use to statistically recover the prior chain context even when the quantization table itself is unsalvageable.

This is reflected in the Step 1 accuracy jumping from **23.9% to 97.0%**, demonstrating that while Discord remains a "trace eraser" for deterministic methods, statistical distribution analysis provides a reproducible forensic recovery path.

---

## Project Structure

```
├── core/                    Feature extractor (272-dim V2) + legacy BKS fusion logic
├── models/                  Pre-trained artifacts (seq_model.joblib)
├── assets/                  Diagnostic graphs
├── samples/                 Ground-truth 2026 test images (1,000 images)
├── scripts/                 export_models.py, reproduce_results_offline.py
├── analyze_image.py         Single-image CLI tool
├── run_all.py               Master pipeline orchestrator (one-click reproducibility)
├── manifest.json            Full 1,000-image ground-truth tracking schema
├── unified_sequence_cache.npz   816-dim chain feature cache
├── samples_test_split.json  Strict holdout split (no parent-child leakage)
├── methodology.md           Data pipeline and feature engineering details
├── research_summary.md      Tracing limitation mathematical explanation
└── requirements.txt
```
