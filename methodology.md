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

We represent each image with a 258-dimensional vector drawn entirely from JPEG block structure. The motivating insight is that platform compression engines consistently alter quantization table values and chroma subsampling, but they cannot undo the spatial *shape* of DCT blocks from a prior compression round.

**Intra-block Markov transitions (T=4, 81 dimensions).** For each 8×8 DCT block, we read AC coefficients in zigzag order and record transitions between quantized integer bins (clipped to T=4 levels). The 9×9 transition probability matrix (flattened to 81 values) captures the local spatial texture in a way that survives quantization changes.

**Ghost peak null-bins (16 dimensions).** When an image is compressed twice, the second quantization step tends to push energy out of specific AC histogram bins, leaving visible null-bins. We measure the density of zero-occupancy bins across coarse, mid, and high-frequency AC positions.

**Q-table L1 distances (6 dimensions).** The mean absolute deviation between the extracted Luma quantization table and each entry in our standard-library table database (2024 and 2026 versions of the three platforms).

---

## Adversarial Pruning — Defeating Concept Drift

Initial experiments showed that standard classifiers trained on 2024 data collapsed to approximately 33% accuracy on 2026 images. The reason was straightforward: the model had learned to use absolute quantization coefficient magnitudes as discriminative features. These values shifted substantially between years (Slack's median Luma coefficient changed by roughly 53%) while remaining stable *within* a given year.

We ran a two-sample Kolmogorov–Smirnov test on each feature dimension across the two yearly datasets and removed any dimension with a significance level below p = 0.05. The remaining dimensions — dominated by Markov transition probabilities — showed strong platform discriminability while exhibiting no statistically significant year-on-year drift. Training on this pruned feature set with a dataset that mixed 2024 and 2026 images produced a Random Forest model with **94.2% single-step platform identification accuracy** on held-out 2026 data.
