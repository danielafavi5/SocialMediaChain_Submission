# Research Summary: From Tracing Limit to Partial Recovery

## Relation to Prior Work

This project extends the framework introduced by Verde, Pasquini, Lago, Goller, De Natale, Piva, and Boato in *Multi-Clue Reconstruction of Sharing Chains for Social Media Images* (IEEE Transactions on Multimedia, vol. 25, 2023). That work proposed a cascade architecture of backtracking blocks — each block being an ensemble of classifiers fused via late fusion — and demonstrated that image sharing chains could be reconstructed up to three steps back across Facebook, Flickr, and Twitter.

This work extends that framework in four directions:

**1. Platform modernisation with live API data collection.** The original used offline JPEG simulators configured to approximate platform compression. For Facebook, Flickr, and Twitter circa 2021, this was a reasonable approximation. For Discord, Slack, and Telegram in 2026, it is not: these platforms do not publish their quantization tables, update their compression pipelines without notice, and produce results that no offline encoder reliably replicates. Every training image in this project was transmitted through the real platform API and captured from the live CDN response.

**2. A deterministic Q-table divisibility layer (archived).** The V1 implementation added a second inference layer operating on the actual Luma quantization table: if the ratio `Q_observed / Q_candidate` was close to a matrix of positive integers, the candidate platform was geometrically confirmed as a prior step. This provided interpretable, measurement-grounded decisions when it fired — but was completely blind to Discord-mediated chains.

**3. Formal characterisation of the Discord tracing limit.** Discord's 2026 Luma quantization table contains coefficients in the range 1–9. These values are too small and too clustered to produce reliable near-integer ratios when used as a divisor against a real compressed image. Once an image passes through Discord, the prior compression history is arithmetically erased from the Q-table domain. This is not a model failure — it is a mathematical boundary of the divisibility-based approach.

**4. Partial recovery via Benford's Law Distribution Analysis and Container Byte Analysis.** The V2 Unified Sequence Engine introduces two additional feature classes that operate outside the Q-table domain, enabling partial signal recovery even beyond the Discord tracing limit.

---

## The Problem

Reconstructing a multi-step image sharing sequence (e.g., Slack → Telegram → Discord) requires tracing backwards through successive JPEG quantization rounds. Each platform re-encodes the image using its own quantization table **Q_k**. The final image on disk carries the coefficient structure of the last platform only; prior encoding rounds leave secondary footprints in the form of structured null-bins and quantization residuals.

We formalized the question: given the final image, can we recover the platform sequence that produced it?

## The Divisibility Heuristic (Archived V1 Approach)

For a two-step chain `A → B`, the ratio matrix **Q_observed / Q_prior** should be close to a matrix of positive integers if **Q_prior** contributed to the current state. We compute:

```
ratio = Q_observed / Q_prior
error = mean( |ratio - round(ratio)| )
```

If `error < 0.25` and the mean ratio value exceeds 1.0, the prior platform is classified as the ghost ancestor. This worked for Telegram/Slack ancestors but was completely defeated by Discord's low-coefficient table.

## The Discord Tracing Limit

Discord's 2026 Luma quantization table contains small, clustered coefficients — starting at 1 for the lowest frequencies and rising gradually to 9 at the highest. When this table is used as the candidate library entry in the divisibility check, the resulting ratios are large and irregular because the observed image has passed through a subsequent platform with substantially larger Q-values. The mean L1 error between those ratios and the nearest integers exceeds the acceptance threshold, so the check returns no match.

The practical consequence — which we call the *Discord tracing limit* — is as follows: **once an image passes through Discord, Discord's compression removes the information needed to recover earlier steps via Q-table arithmetic.** No mathematical formula operating on quantization table ratios can look past this limit.

**Why the original paper's cascade approach would not solve this.** The backtracking block cascade proposed by Verde et al. uses ensembles of classifiers fed multiple feature representations — DCT statistics, metadata flags, and container-level structural features. All of those features are derived from the image that remains *after* the most recent platform has processed it. For any chain passing through Discord, the relevant prior-platform signal is compressed into Discord's low-coefficient space and lost. A more sophisticated classifier ensemble operating on the same post-Discord image would face the same information gap.

## Benford's Law Distribution Analysis — Recovery Beyond the Horizon

Benford's Law states that in many naturally occurring numerical datasets, the leading digit `d` appears with frequency:

```
P(d) = log10(1 + 1/d)     for d ∈ {1, 2, ..., 9}
```

This means digit 1 appears ~30.1% of the time, digit 2 ~17.6%, and digit 9 only ~4.6%. Natural image DCT coefficient populations approximately follow this distribution. When a JPEG image is compressed, the quantization step truncates coefficient values in a platform-specific pattern. When a second platform then re-quantizes the result, the leading-digit distribution of the surviving AC coefficients is perturbed in a characteristic, reproducible way — the perturbation pattern depends on the *ratio* of the two quantization tables.

Critically, **this perturbation pattern is not destroyed by Discord's re-quantization in the same way that divisibility ratios are.** While Discord's small Q-values do collapse the absolute coefficient magnitudes, they produce a characteristic redistribution of leading digits — specifically, a concentration of power toward digit 1 and an anomalous suppression of digits 3–7 — that persists in the coefficient distribution even after platform re-quantization. The 9-bin Benford probability array we extract from AC coefficients encodes this redistribution as a platform-attributable fingerprint.

This mechanism allows the Unified Sequence Engine to partially recover chain context even when the Q-table divisibility signal is entirely destroyed. It does not recover the full prior-platform identity with certainty — Discord's transformation is still lossy — but it provides a statistically discriminative signal that the Random Forest can use alongside the container marker structure to make a well-informed probabilistic guess.

## Container Byte Analysis — Platform Backend Fingerprints

Separate from coefficient statistics, each JPEG file contains a binary header composed of fixed-format marker sequences. The order and presence of these markers (APP0, APP1, DQT, DHT) depends on the libjpeg backend library used by the platform server to reconstruct the file. Because platform backends are consistent across uploads from the same service, these structural flags function as a digital fingerprint that is entirely independent of image content and immune to re-quantization.

Discord, Telegram, and Slack differ in their characteristic marker orderings — specifically in whether the DHT (Huffman table) block appears before or after the DQT (quantization table) block. This single binary flag, along with the presence or absence of APP0/APP1 application markers, is a reproducible platform-attributable signal that survives even across the Discord compression boundary.

---

## Results Overview

### V2 Unified Sequence Engine (current: `master`)

| Task | Metric | Score |
|---|---|---|
| Step 1 platform reconstruction | chain accuracy | **74.6%** |
| Step 2 platform reconstruction | chain accuracy | **53.7%** |
| Step 3 platform reconstruction (surface) | chain accuracy | **55.2%** |
| Exact 3-step chain reconstruction | chain accuracy | **32.8%** |

*Evaluated on 67 held-out chains, spanning all 6 permutations of Telegram, Slack, and Discord.*

### V1 Modular Architecture (reference: `archive/v1-modular-baseline`)

| Task | Metric | Score |
|---|---|---|
| C1: single-step surface identification | macro-F1 | **66.5%** |
| C2: prior platform identification (residual) | macro-F1 | **31.2%** |
| BKS fused 3-step chain reconstruction | chain accuracy | **4.5%** |

### Improvement Summary

| Metric | V1 | V2 | Change |
|---|---|---|---|
| Step 1 accuracy | 23.9% | 74.6% | **+50.7pp** |
| Exact 3-step match | 4.5% | 32.8% | **+28.3pp** |

The 32.8% exact sequence match represents a **7.3× improvement** over the prior BKS fused baseline. The step 1 accuracy improvement from 23.9% to 74.6% directly demonstrates the effectiveness of Benford's Law Distribution Analysis and Container Byte Analysis in providing residual signal beyond the Discord tracing limit.

---

## Conclusion

The Discord tracing limit — a mathematical boundary imposed by Discord's aggressive low-coefficient quantization — is not fully overcome by the V2 approach. It remains the case that Q-table divisibility cannot recover any prior-compression information once Discord has processed an image. However, the Unified Sequence Engine demonstrates that two alternative signal classes — Benford's Law coefficient distribution analysis and JPEG container byte ordering — provide statistically meaningful partial recovery that was not accessible to the V1 modular approach.

The architectural shift from independent per-step classification (C1/C2/BKS) to a unified 816-dimensional chain signature with simultaneous multi-output prediction is the primary driver of the accuracy improvement. By treating the full compression history as a single structured input rather than a sequence of isolated events, the model is able to leverage cross-step dependencies that would otherwise be discarded.

Future work that could further reduce the horizon effect would involve generative AI-based signal reconstruction (inferring the pre-Discord coefficient distribution from context) or CNN-based spatial analysis that operates on the raw pixel grid rather than global statistical summaries.
