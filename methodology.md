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

We represent each image with a 258-dimensional vector drawn entirely from JPEG block structure and metadata parameters. The motivating insight is that platform compression engines consistently alter quantization table values and chroma subsampling, but they cannot undo the spatial *shape* of DCT blocks from a prior compression round.

**DCT AC Histograms & Energy Maps (42 dimensions).** We measure the statistical distribution and absolute energy of the first 21 AC coefficients in zigzag order.

**Intra-block Markov transitions (T=4, 81 dimensions).** For each 8×8 DCT block, we read AC coefficients in zigzag order and record transitions between quantized integer bins (clipped to T=4 levels). The 9×9 transition probability matrix (flattened to 81 values) captures the local spatial texture in a way that survives quantization changes.

**Luminance & Chrominance Q-Tables + Stats (112 dimensions).** We directly embed the 64-element Luma table and the 40-element Chroma table, alongside mean, std, min, and max statistical aggregations for both.

**Metadata & Structural Flags (12 dimensions).** These dimensions capture explicit properties parsed from the header (e.g., progressive encoding, quality estimate flags).

**Ghost peak null-bins (5 dimensions).** When an image is compressed twice, the second quantization step tends to push energy out of specific AC histogram bins, leaving visible null-bins. We measure the density of zero-occupancy bins exclusively across the critical AC1-AC5 high-frequency positions.

**Q-table L1 distances (6 dimensions).** The mean absolute deviation between the extracted Luma quantization table and each entry in our standard-library database (2024 and 2026 versions of Telegram, Discord, Slack).

---

## Robust Feature Selection — Universal Forensic Invariants

Initial experiments showed that classifiers trained exclusively on 2024 legacy platforms (Facebook, Flickr, Twitter) struggled to generalize to 2026 target platforms (Telegram, Slack, Discord). The core challenge was distinguishing between **platform-specific discriminants** and **temporal instability**.

To identify the most reliable features, we performed a two-sample Kolmogorov–Smirnov (KS) test across the two yearly datasets. Even though the specific platforms differed between 2024 and 2026, this test allowed us to isolate **Universal Forensic Invariants**—features like Intra-block Markov transitions that remain statistically stable across diverse encoding engines and timeframes. By pruning dimensions that exhibited high variance across this heterogeneous historical set, we ensured the model focuses on structural JPEG artifacts rather than absolute quantization magnitudes. Training on this refined feature set produced a Random Forest model with **66.0% single-step platform identification macro-F1** across the 2026 target set.

---

## Ghost Simulation and Data Segregation

The residual classifier (C2) is trained on intermediate chain steps — the step 1 and step 2 images that were passed through a real platform before being passed to the next one. Two design decisions ensure this training data reflects real-world conditions rather than artificially clean inputs.

**Ghost Simulation.** Before extracting features from a training image, the pipeline re-saves it as a JPEG at quality 75 using PIL into a temporary file, extracts the 258-dimensional feature vector from that file, and then deletes it. This ensures the features C2 learns from represent a signal that has already been through at least one compression round — matching the actual condition of intermediate chain images at inference time.

**Chain-ID segregation.** Each image group in the dataset is assigned a `chain_id` derived from the SHA-256 of the payload at upload time. When building the C2 training set, the pipeline reads the 20% holdout file list from `samples_test_split.json`, extracts the `chain_id` of every image in that list, and removes any training sample whose `chain_id` appears in that set. This guarantees that no training example comes from the same source image as any test example, even across different compression steps.

Together, these two steps ensure C2 is evaluated on genuinely unseen data and learns compression residue patterns rather than surface-level artifacts from pristine originals.
