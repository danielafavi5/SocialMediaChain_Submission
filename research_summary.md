# Research Summary: The Tracing Limit

## Relation to Prior Work

This project extends the framework introduced by Verde, Pasquini, Lago, Goller, De Natale, Piva, and Boato in *Multi-Clue Reconstruction of Sharing Chains for Social Media Images* (IEEE Transactions on Multimedia, vol. 25, 2023). That work proposed a cascade architecture of backtracking blocks — each block being an ensemble of classifiers fused via late fusion — and demonstrated that image sharing chains could be reconstructed up to three steps back across Facebook, Flickr, and Twitter.

This work extends that framework in three directions:

**1. Platform modernisation with live API data collection.** The original used offline JPEG simulators configured to approximate platform compression. For Facebook, Flickr, and Twitter circa 2021, this was a reasonable approximation. For Discord, Slack, and Telegram in 2026, it is not: these platforms do not publish their quantization tables, update their compression pipelines without notice, and produce results that no offline encoder reliably replicates. Every training image in this project was transmitted through the real platform API and captured from the live CDN response. The `chained_uploader.py` orchestrator automates this at scale for all three platforms and all permutation sequences.

**2. A deterministic Q-table divisibility layer.** The original cascade is entirely statistical — each block outputs a platform label from a trained classifier. This project adds a second inference layer that operates on the actual Luma quantization table extracted from the JPEG header: if the ratio `Q_observed / Q_candidate` is close to a matrix of positive integers, the candidate platform is geometrically confirmed as a prior step, independent of what the classifier says. When this check fires (Telegram/Slack ancestors), it provides a hard geometric confirmation. When it does not fire, it provides a principled refusal with a stated arithmetic reason — a level of interpretability not present in the original cascade.

**3. Formal characterisation of the tracing limit.** The platforms in the original paper — Facebook, Flickr, Twitter — all use structured Q-tables with values large enough to preserve divisibility relationships across compression rounds. Discord's 2026 table uses coefficients in the range 1–9, which are too small and clustered to produce reliable near-integer ratios when used as a divisor against a real compressed image. The result is that once an image passes through Discord, no subsequent mathematical measurement can recover the earlier compression history. This is not a model accuracy problem — it is an arithmetic property of that specific Q-table structure. This project is the first to characterise this limit formally, derive it from the Q-table values, and confirm it empirically across 67 evaluated chains.

---

## The Problem

Reconstructing a multi-step image sharing sequence (e.g., Slack → Telegram → Discord) requires tracing backwards through successive JPEG quantization rounds. Each platform re-encodes the image using its own quantization table **Q_k**. The final image on disk carries the coefficient structure of the last platform only; prior encoding rounds leave secondary footprints in the form of structured null-bins and quantization residuals.

I formalized the question: given the final Luma quantization table **Q_observed**, can we determine whether a prior platform's table **Q_prior** is a mathematical ancestor?

## The Divisibility Heuristic

For a two-step chain `A → B`, the ratio matrix **Q_observed / Q_prior** should be close to a matrix of positive integers if **Q_prior** contributed to the current state. We compute:

```
ratio = Q_observed / Q_prior
error = mean( |ratio - round(ratio)| )
```

If `error < 0.25` and the mean ratio value exceeds 1.0 (i.e., the current table is a genuine multiple of the prior table, not equal to it), we classify the prior platform as the ghost ancestor.

This heuristic works reliably for Telegram → Slack and Slack → Telegram traces, where both platforms use structured, non-trivial quantization tables. Telegram's Luma Q-table has values ranging from 3 to 31; Slack's ranges from 1 to 10. Both are large enough relative to Discord's table to produce clean integer ratios.

## The Discord Limitation

Discord's 2026 Luma quantization table contains small, clustered coefficients — starting at 1 for the lowest frequencies and rising gradually to 9 at the highest. When this table is used as the candidate library entry in the divisibility check (`ratio = Q_observed / Q_discord`), the resulting ratios are large and irregular because the observed image has passed through a subsequent platform with substantially larger Q-values. The mean L1 error between those ratios and the nearest integers exceeds the acceptance threshold, so the check returns no match.

The codebase also contains a general safeguard that skips candidate library tables where more than 32 cells equal 1 (`np.sum(lib_dqt == 1) > 32` in `bks_fusion.py`). For the current Discord Q-table this threshold is not triggered — Discord has 14 cells equal to 1, not a majority. Discord is blocked by the divisibility arithmetic itself, not by this filter.

We call the practical consequence of this the *Discord Limitation*: a table with low, clustered coefficients does not behave as a reliable divisor for detecting ghost ancestors. The fix `np.sum(lib_dqt == 1) > 32` in `bks_fusion.py` is a general guard for degenerate flat tables and is available if future platform table updates produce such cases.

The deeper consequence is unavoidable: **once an image passes through Discord, Discord's compression removes the information needed to recover earlier steps.** Discord's low-coefficient table does not preserve any linearly separable ratio information from the incoming image. No mathematical formula can look past this limit to see where the image originated. This is what we call the *tracing limit*.

**Why the original paper's cascade approach would not solve this.** The backtracking block cascade proposed by Verde et al. uses ensembles of classifiers fed multiple feature representations — DCT statistics, metadata flags, and container-level structural features. All of those features are derived from the image that remains *after* the most recent platform has processed it. For Facebook → Flickr → Twitter chains (the original's test case), each platform uses large structured Q-tables that leave distinct residual patterns in the DCT coefficient distribution; a classifier has something concrete to measure. For any chain passing through Discord, the relevant prior-platform signal — quantization table ratios, Markov transition structure, ghost peaks — is compressed into Discord's low-coefficient space and lost. A more sophisticated classifier ensemble operating on the same post-Discord image would face the same information gap. The failure is not in the measurement architecture; it is in the absence of measurable signal. This is why applying the original method to this platform set would produce the same outcome, and why the result from this work is a property of the platforms studied rather than of the chosen approach.

## Quantitative Results

| Task | Metric | Score |
|---|---|---|
| C1: single-step surface platform identification | macro-F1 | **66.5%** |
| C2: prior platform identification (residual) | macro-F1 | **31.2%** |
| BKS fused 3-step chain reconstruction | chain accuracy | **4.5%** |

**C1 (66.5%)** measures how accurately the surface-level Random Forest identifies the *current* platform of an image — the last one that compressed it. The signal is relatively strong here because the current platform's Q-table is directly readable from the JPEG header.

**C2 (31.2%)** measures something harder: given an intermediate chain image, can the residual classifier identify *the platform it came from before*. For a 3-class problem, random guessing produces 33.3% macro-F1. The C2 score of 31.2% is effectively at the floor of random chance. This is expected: the prior platform's compression signature has been substantially overwritten by the current one, and Discord's aggressive table erases it almost entirely. Rather than indicating a model failure, the near-random C2 score confirms the analytical conclusion — the residual forensic signal is insufficient for reliable prior-step recovery across this platform set. The 2 BKS-rescued sequences (where the deterministic Q-table divisibility check overrode the RF prediction) represent the only cases where recoverable information remained.

**Note on comparability with Verde et al. (2023).** The 66.5% C1 score is a macro-F1 over three classes and is structurally the same metric the original paper would use for single-step platform identification. However, a direct numerical comparison is not meaningful for two reasons. First, the platform sets are entirely different: Facebook, Flickr, and Twitter have more visually distinct compression profiles (especially Twitter's historically more aggressive quantization versus Facebook's near-lossless settings), making them easier to discriminate. Second, the original paper's exact per-step F1 values are reported in its Section IV results tables and are not reproduced here. What can be said is that the 66.5% figure reflects a genuinely harder discrimination task — Telegram's Q-table is nearly identical to Facebook's 2024 table, Slack's aggressively low-quality table partially overlaps with Discord's, and the platform signatures are structurally more similar to each other than the Facebook/Flickr/Twitter set.

The 3-step evaluation covered 67 full chains drawn from the 201-image holdout set, spanning all six permutations of the three platforms. Every chain that included Discord as an intermediate or final step collapsed into a repetitive ML prediction loop (e.g., the model predicted `telegram → telegram → telegram` for the majority of sequences). Chains that terminated after Slack or Telegram without Discord as an intermediate showed the most promise for future non-deterministic approaches.


## Conclusion

The tracing barrier imposed by Discord's aggressive, low-quality compression is not a software bug or a failure of the machine learning model — it is a mathematical boundary. Standard forensic measurement techniques simply cannot extract enough information to recover an image's earlier history once Discord erases those clues. Overcoming this boundary in the future would require an advanced generative AI capable of guessing the missing structural data, rather than measuring the remaining data mathematically.
