# Visual Assets for Model Evaluation

This directory contains the visual representations generated from the evaluation of the Unified Sequence Engine (V2) pipeline.

---

## 1. System and Feature Structure

*   **`pipeline_flowchart.png`**: **Pipeline Diagram**. A block diagram showing the end-to-end processing sequence: Data Ingestion $\rightarrow$ Feature Extraction (272 dimensions) $\rightarrow$ Sequential Classifier Chain $\rightarrow$ Behavior-Knowledge Space (BKS) traceback $\rightarrow$ Output Sequence.
*   **`feature_composition.png`**: **Feature Vector Composition**. A block diagram showing the dimensions and index slices of the 11 forensic feature families extracted from each image, as defined in `core/forensic_features.py`.

---

## 2. Evaluation Results

*   **`feature_importance.png`**: **Feature Family Importance**. A bar chart showing the aggregate Random Forest feature importances grouped by feature family.
*   **`offline_evaluation_metrics.png`**: **Holdout Performance Metrics**. A bar chart showing the exact sequence match accuracy (87.5%) and per-step classification accuracies (Step 1: 97.2%, Step 2: 88.9%, Step 3: 88.9%) on the held-out test split.
*   **`model_comparison.png`**: **Model Comparison**. A bar chart comparing the exact sequence match accuracy of the finalized sequence model (87.5%) and the experimental 2026 model (84.7%) on the held-out test split.
*   **`sequence_confusion_matrix.png`**: **Sequence Confusion Matrix**. A row-normalized confusion matrix heatmap of predicted vs. true sharing sequences on the test split.
*   **`example_predictions_table.png`**: **Sample Predictions Table**. A table displaying a subset of the first 10 testing samples, including the Sample ID, ground truth sequence, predicted sequence, and binary match flag.
