# Methodology

## Dataset Acquisition — Live Platform Chaining

A central engineering constraint of this project was that platform compression algorithms are proprietary, undocumented, and change silently between software releases. No offline simulation — whether PIL, OpenCV, or standard libjpeg — can faithfully reproduce the quantization table choices, chroma subsampling decisions, or metadata-stripping behavior of a live server.

To obtain authentic 2026 training data, I wrote `core/chained_uploader.py`, a Python orchestrator that transmits images through real Discord, Telegram, and Slack APIs in sequence and captures the server-returned JPEG bytes directly from each platform's CDN.

## Source Material — The Pristine Baseline

To ensure the integrity of our ground-truth labels, I did not use images found on the web. Instead, I pulled 3,000 high-resolution, uncompressed images from two recognized forensic benchmarks: RAISE (Research All-around Image Dataset) and VISION. By starting with these "pristine" files, I guaranteed that any forensic traces discovered later were caused exclusively by my chained_uploader and not by unknown historical compression.

**How one chaining step works:**

1. The script wraps the source image bytes in the appropriate SDK object (`discord.File`, `bot.send_photo`, or `files_upload_v2`).
2. It waits for the server to acknowledge the upload and process the attachment.
3. It fetches the re-encoded file from the platform's CDN via an authenticated HTTP GET call and stores the raw bytes in memory.
4. Those bytes become the *input* for the next platform in the sequence, mathematically stacking the compression degradation.

**The 7.9 MB safeguard.** Platforms enforce payload limits (Slack will raise an `IncompleteRead` error for files above ~4 MB; Discord caps file uploads at 8 MB). To keep image payloads within limits without introducing artifacts from an arbitrary third-party compression step, I implemented a minimal PIL degradation loop:

```python
quality = 98
while quality > 10:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    if buf.tell() <= max_size:
        break
    quality -= 2
```

Each iteration reduces quality by only 2 points and checks in memory before writing anything. The first quality level that satisfies the byte budget is transmitted, allowing the platform server to apply its own final re-compression pass. This keeps the pre-platform encoding artifact as small as possible.

**Ground-truth labeling.** Every transmitted image is saved alongside a `manifest.json` entry containing a `chain_id` (the first 8 hex characters of the SHA-256 hash of the payload at transmission time), the exact platform sequence, and the step index. Filenames also encode this: `{image}.chain_{id}.step{N}.{platform}.jpg`. This two-layer traceability ensures the evaluation script and the actual file on disk can never be de-synchronized.

---

## Feature Extraction — Structural Invariants Over Metadata

We represent each image with a **272-dimensional vector** drawn entirely from JPEG block structure, quantization parameters, coefficient statistics, and container-level byte markers. The motivating insight is that platform compression engines consistently alter quantization table values and chroma subsampling, but they cannot undo the spatial *shape* of DCT blocks from a prior compression round.

**DCT AC Histograms & Energy Maps (42 dimensions).** We measure the statistical distribution and absolute energy of the first 21 AC coefficients in zigzag order.

**Intra-block Markov transitions (T=4, 81 dimensions).** For each 8×8 DCT block, we read AC coefficients in zigzag order and record transitions between quantized integer bins (clipped to T=4 levels). The 9×9 transition probability matrix (flattened to 81 values) captures the local spatial texture in a way that survives quantization changes.

**Luminance & Chrominance Q-Tables + Stats (112 dimensions).** We directly embed the 64-element Luma table and the 40-element Chroma table, alongside mean, std, min, and max statistical aggregations for both.

**Metadata & Structural Flags (12 dimensions).** These dimensions capture explicit properties parsed from the header (e.g., progressive encoding, quality estimate flags).

**Ghost peak null-bins (5 dimensions).** When an image is compressed twice, the second quantization step tends to push energy out of specific AC histogram bins, leaving visible null-bins. We measure the density of zero-occupancy bins exclusively across the critical AC1-AC5 high-frequency positions.

**Q-table L1 distances (6 dimensions).** The mean absolute deviation between the extracted Luma quantization table and each entry in our standard-library database (2024 and 2026 versions of Telegram, Discord, Slack).

**Benford's Law Distribution Analysis (9 dimensions).** We analyze the leading-digit probability distribution of all non-zero high-frequency AC DCT coefficients. Under natural image statistics, the frequency of leading digits follows Benford's Law (digit 1 appears ~30% of the time, digit 9 approximately 5%). Each successive platform re-quantization perturbs this distribution in a characteristic, platform-dependent direction. These 9 probability bins provide a non-divisibility fingerprint that remains partially legible even after Discord's aggressive low-coefficient quantization, because the distribution's shape encodes prior-round structure that is independent of absolute coefficient magnitudes.

**Container Byte Analysis (5 dimensions).** The JPEG file format encodes metadata in a sequence of binary markers. Different platforms reconstruct the JPEG container using different backend libjpeg variants, producing characteristic differences in the ordering of markers such as APP0 (`\xff\xe0`), APP1 (`\xff\xe1`), DHT (`\xff\xc4`) and DQT (`\xff\xdb`). We extract five binary flags encoding the presence and relative ordering of these markers. Because these flags reflect server-backend choices rather than image content, they survive even aggressive re-quantization and are immune to the quantization erasure problem that limits table-divisibility approaches.

---

## Robust Feature Selection — Universal Forensic Invariants

Initial experiments showed that classifiers trained exclusively on 2024 legacy platforms (Facebook, Flickr, Twitter) struggled to generalize to 2026 target platforms (Telegram, Slack, Discord). The core challenge was distinguishing between **platform-specific discriminants** and **temporal instability**.

To identify the most reliable features, we performed a two-sample Kolmogorov–Smirnov (KS) test across the two yearly datasets. Even though the specific platforms differed between 2024 and 2026, this test allowed us to isolate **Universal Forensic Invariants** — features like Intra-block Markov transitions that remain statistically stable across diverse encoding engines and timeframes. By pruning 15 dimensions that exhibited high cross-year variance, we ensured the model focuses on structural JPEG artifacts rather than absolute quantization magnitudes. This adversarial pruning is applied uniformly across all three step positions in the 816-dimensional chain signature.

---

## Architecture — Unified Sequence Classification

### Archived Modular Architecture (reference: `archive/v1-modular-baseline`)

The original implementation followed a cascade architecture with three independent components:

- **C1 (Surface Classifier):** A Random Forest trained to identify the *current* platform from a single image's 258-dimensional feature vector.
- **C2 (Residual Classifier):** A second Random Forest trained on intermediate chain images to identify the *prior* platform.
- **BKS Fusion:** A deterministic Q-table divisibility heuristic layer in `core/bks_fusion.py` that attempted to override the C2 prediction when a clean integer-ratio relationship was detected between the observed Luma table and a candidate library entry.

This modular approach treated each step of the platform chain as an **independent classification event**. The BKS fusion layer attempted to post-hoc reconnect these independent predictions, but because the individual C1 and C2 models were optimized separately and the Q-table divisibility check could not operate on the post-Discord signal at all, the end-to-end 3-step chain accuracy was limited to approximately 4.5%.

### Unified Sequence Engine (current: `master`)

The current architecture replaces the modular C1/C2/BKS stack with a **single MultiOutput Sequence Classifier** that treats the platform sharing chain as a **dependent sequential timeline**.

**Training input:** For each chain in the training set, we concatenate the 272-dimensional feature vectors of Step 1, Step 2, and Step 3 images into a single **816-dimensional chain signature**. This ensures the model receives the complete compression history at once, rather than making per-step predictions in isolation.

**Training target:** The model simultaneously predicts `[platform_step1, platform_step2, platform_step3]` as a 3-element output. The Scikit-Learn `MultiOutputClassifier` wrapper trains one Random Forest per output dimension, but crucially the input to all three forests is the same 816-dimensional signal — enabling each output to implicitly condition on the full inter-step dependency structure encoded in the concatenated features.

**Adversarial pruning:** The same 15 drift-prone feature indices identified by the KS test are masked out from all three 272-dimensional blocks within the 816-dimensional vector (45 features removed in total), maintaining the anti-leakage guarantees established in the original architecture.

**Result:** This unified treatment raised the exact 3-step chain reconstruction accuracy from **4.5%** (modular BKS fused) to **32.8%**, with Step 1 accuracy reaching **74.6%** — demonstrating that the inter-step dependency information encoded in the full chain signature is substantially more informative than independent per-step classification.
