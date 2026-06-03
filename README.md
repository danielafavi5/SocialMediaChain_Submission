# TrueFake-IJCNN25 — Forensic Platform Identification

Reproduction package for the study:  
*Multi-Clue Reconstruction of Sharing Chains for Social Media Images*  
University of Trento — Multimedia Data Security 2026

---

> [!IMPORTANT]
> **Context for Evaluators: Diagnostic vs. Real-World Forensics**
> This repository contains two distinct analysis paradigms:
> 1. **Theoretical Diagnostic (`reproduce_results_offline.py`):** This script concatenates all 3 images in a chain into an 816-dimensional signature to test the absolute mathematical upper-bound of sequence recovery (yielding **87.5%** accuracy). It assumes full access to the image history (Step 1 + 2 + 3).
> 2. **Real-World Forensic Tool (`analyze_image.py`):** In a real investigation, only the final image is available. This CLI tool strictly adheres to this limitation. It operates on a single image (272 dims) to predict the surface platform and utilizes a BKS divisibility heuristic to recover the immediate ghost prior. Because of the mathematical "Discord Trace Eraser" limit (detailed in the research summary), single-image analysis is strictly capped at a 2-step history.


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

## Active Holdout Set Results (Unified V2)

Evaluating the model strictly on the leak-free, blind testing split (`samples_test_split.json`) consisting of 72 chains, the offline evaluation script computes both evaluation modes:

| Metric | Concatenated (Validation Sandbox) | True Blind (Tracing Limit) |
| :--- | :---: | :---: |
| **Step 1 Platform Accuracy** | **97.2%** | **50.0%** |
| **Step 2 Platform Accuracy** | **88.9%** | **29.2%** |
| **Step 3 Platform Accuracy** | **88.9%** | **55.6%** |
| **Exact 3-Step Match Accuracy** | **87.5%** | **11.1%** |

*Note on Evaluation Leakage*: The concatenated validation scores reflect a **Diagnostic Validation Check** using concatenated 816-dimensional features from all three images (Step 1, Step 2, and Step 3). Under a practical, blind forensic scenario where only the final Step 3 image is available (no intermediate features), the model is limited to the **True Blind (Tracing Limit)** accuracy of **11.1%** due to lossy signature overwriting.

### Platform Tracing Limitations & Fallbacks

*   **API Sandbox Behavior & Telegram Dominance**: In the live-API sandbox, Discord serves attachments losslessly. Because Discord is lossless and Slack is lossless for standard-sized images, **Telegram's forced server-side compression dominates the entire chain**. The final Step 3 image always carries Telegram's Q-table, resulting in a zero-variance Q-table feature set at Step 3. Whether due to Discord's compression in the real world or lossless propagation in our sandbox, traditional Q-table divisibility fails completely on the final step.
*   **CLI Predictor Fallback**: Because the Classifier Chain sequence model (`seq_model.joblib`) requires the 816-dimensional concatenated vector of three distinct images, the standalone CLI tool (`analyze_image.py`) cannot run it on a single user-supplied image. The CLI tool is therefore decoupled from the sequence chain and falls back to a 2-step single-image pipeline: predicting the surface platform using `surface_model.joblib` and back-tracking a single prior step using BKS.
*   **BKS Non-Trivial Divisor Masking**: Divisibility traceback heuristics are structurally biased towards low-coefficient Q-tables (like Slack or Discord), as division by small integers (such as 1s and 2s) naturally yields near-integer ratios. To eliminate this bias, we implement **Non-Trivial Divisor Masking**, calculating the L1 divisibility error strictly on coefficients where the candidate prior table contains values greater than 1, requiring at least 8 non-trivial bins to validate the trace.

### How Benford's Law Distribution Analysis Partially Recovers Traces

The V2 architecture overcomes these Q-table limits via **Benford's Law analysis** of AC DCT coefficients. Successor platform re-quantization introduces deviations in the expected leading-digit distribution of AC coefficients (where natural signals follow Benford's Law). These distribution anomalies — combined with JPEG container-level marker order fingerprints (`has_dht`, `dqt_before_dht`) — provide a non-divisibility signal that the Random Forest uses to statistically recover the prior chain context.

This is reflected in the Step 1 accuracy jumping from **23.9% to 97.2%**, demonstrating that while Discord remains a trace eraser for deterministic methods, statistical distribution analysis provides a reproducible forensic recovery path.

### Dataset Scalability Discovery
To demonstrate that the Unified Engine natively scales with more data, we have designed two supplementary test scripts to run on a larger ~7,000 image dataset. 

> [!NOTE]
> **Dataset Download:** Due to Git file size limits, the 11 GB dataset is hosted externally. You can download the images from [this Google Drive link](https://drive.google.com/file/d/1Ozet6Iqi9qUw-lLKO3iO7HgztYXB-5x6/view?usp=sharing). To run the scalability scripts below, please download the archive and extract it such that the images are placed in a folder named exactly `results_2026/` within the root directory of this repository.

The included scalability testing scripts are:
- **`scripts/test_train_2026.py`**: Trains the Unified Engine strictly on the large 2026 dataset (using `GroupShuffleSplit` on the source image to prevent data leakage) and saves the resulting model as `models/seq_model_2026.joblib`.
- **`scripts/compare_models.py`**: Performs a strict apples-to-apples comparison by evaluating both the original subset-trained model and the new 2026 model against the exact same 67-chain blind holdout set (`samples_test_split.json`).

**Results:** The original subset-trained model achieved **87.5%** exact sequence match, while the experimental 2026 model achieved **84.7%** on the true holdout test set, proving that our base pipeline is extremely robust and the larger dataset did not necessarily improve generalization in this specific configuration.

---

## Project Structure

```
├── core/                    Feature extractor (272-dim V2) + legacy BKS fusion logic
├── models/                  Pre-trained artifacts (seq_model.joblib)
├── assets/                  Diagnostic graphs
├── samples/                 Ground-truth 2026 test images (1,000 images)
├── results_2026/            Larger 7,000-image dataset for scalability testing (Downloaded externally)
├── scripts/                 export_models.py, reproduce_results_offline.py, test_train_2026.py, compare_models.py
├── analyze_image.py         Single-image CLI tool
├── run_all.py               Master pipeline orchestrator (one-click reproducibility)
├── manifest.json            Full 1,000-image ground-truth tracking schema
├── unified_sequence_cache.npz   816-dim chain feature cache
├── samples_test_split.json  Strict holdout split (no parent-child leakage)
├── methodology.md           Data pipeline and feature engineering details
├── research_summary.md      Tracing limitation mathematical explanation
└── requirements.txt
```
