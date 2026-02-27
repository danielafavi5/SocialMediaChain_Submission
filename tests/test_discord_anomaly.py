"""
tests/test_discord_anomaly.py
==============================
Isolated verification of the Discord Anomaly and divisibility logic.
Uses the existing production code paths unchanged.
Run from the repo root:
    python tests/test_discord_anomaly.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.forensic_features import Q_TABLE_LIBRARY
from core.bks_fusion import SequenceAwareBKS

bks = SequenceAwareBKS(Q_TABLE_LIBRARY)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    tag = "PASS" if condition else "FAIL"
    results.append((tag, label))
    print(f"  [{status}] {label}")
    if detail:
        print(f"         → {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Print the raw Discord and Slack Q-tables from the library
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*66)
print("  SECTION 1: Q-Table Inspection")
print("="*66)

discord_qt = Q_TABLE_LIBRARY["2026 Discord"]
slack_qt   = Q_TABLE_LIBRARY["2026 Slack"]
tg_qt      = Q_TABLE_LIBRARY["2026 Telegram"]

discord_ones = int(np.sum(discord_qt == 1))
slack_ones   = int(np.sum(slack_qt == 1))
tg_ones      = int(np.sum(tg_qt == 1))

print(f"\n  Discord 2026 Q-table  : {discord_qt[:16]} ...")
print(f"  Cells equal to 1      : {discord_ones} / 64")
print(f"\n  Slack 2026 Q-table    : {slack_qt[:16]} ...")
print(f"  Cells equal to 1      : {slack_ones} / 64")
print(f"\n  Telegram 2026 Q-table : {tg_qt[:16]} ...")
print(f"  Cells equal to 1      : {tg_ones} / 64")

IDENTITY_THRESHOLD = 32
check(
    "Discord Q-table is identity-like (cells==1 > threshold)",
    discord_ones > IDENTITY_THRESHOLD,
    f"Discord has {discord_ones} ones, threshold={IDENTITY_THRESHOLD}"
)
check(
    "Slack Q-table is NOT identity-like",
    slack_ones <= IDENTITY_THRESHOLD,
    f"Slack has {slack_ones} ones"
)
check(
    "Telegram Q-table is NOT identity-like",
    tg_ones <= IDENTITY_THRESHOLD,
    f"Telegram has {tg_ones} ones"
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Verify _check_pair correctly rejects Discord as a lib_dqt
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*66)
print("  SECTION 2: _check_pair — Identity Filter Behaviour")
print("="*66)

# Simulate image that has gone through Slack, then Discord
# After two compressions, the observed Q-table should resemble Discord's
# (Discord overwrites), so current_dqt ≈ discord_qt
simulated_observed = discord_qt.copy()

matched_discord, meta_discord = bks._check_pair(simulated_observed, discord_qt)
print(f"\n  _check_pair(observed≈discord, lib=Discord)")
print(f"  matched={matched_discord}, meta={meta_discord}")
check(
    "_check_pair returns False for Discord lib_dqt (identity filter)",
    not matched_discord,
    "Discord table has >32 ones, filter applied before ratio computation"
)

matched_slack, meta_slack = bks._check_pair(simulated_observed, slack_qt)
print(f"\n  _check_pair(observed≈discord, lib=Slack)")
print(f"  matched={matched_slack}, meta={meta_slack}")
# Discord's table divided by Slack's non-trivial table will NOT be near-integer
check(
    "_check_pair correctly rejects Discord-observed vs Slack-lib (no clean multiple)",
    not matched_slack,
    f"mean_error={meta_slack['mean_error']:.4f} if matched, otherwise None" if meta_slack else "meta=None as expected"
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Verify _check_pair correctly detects a REAL ancestor (Slack → Telegram)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*66)
print("  SECTION 3: _check_pair — Valid Ghost Detection")
print("="*66)

# To simulate Slack compressed THEN Telegram re-compressed:
# The Telegram Q-table should be a near-integer multiple of Slack's
ratio_tg_over_sl = tg_qt / np.where(slack_qt == 0, 1e-6, slack_qt)
print(f"\n  Telegram/Slack ratio (first 16): {np.round(ratio_tg_over_sl[:16], 2)}")
print(f"  Mean ratio                      : {np.mean(ratio_tg_over_sl):.4f}")
print(f"  Mean |ratio - round(ratio)|     : {np.mean(np.abs(ratio_tg_over_sl - np.rint(ratio_tg_over_sl))):.4f}")

matched_tg_sl, meta_tg_sl = bks._check_pair(tg_qt, slack_qt)
print(f"\n  _check_pair(current=Telegram Q-table, lib=Slack Q-table)")
print(f"  matched={matched_tg_sl}, meta={meta_tg_sl}")
check(
    "_check_pair detects Telegram as multiple of Slack (ghost ancestor)",
    matched_tg_sl,
    f"mean_error={meta_tg_sl['mean_error']:.4f}, mean_nearest={meta_tg_sl['mean_nearest']:.4f}" if meta_tg_sl else "NOT DETECTED"
)
if meta_tg_sl:
    check(
        "ghost meta dict has required keys (ratio, mean_error, mean_nearest)",
        all(k in meta_tg_sl for k in ("ratio", "mean_error", "mean_nearest")),
        str(list(meta_tg_sl.keys()))
    )
    check(
        "ratio is a JSON-serializable list",
        isinstance(meta_tg_sl["ratio"], list),
        f"type={type(meta_tg_sl['ratio']).__name__}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. High-level check_dqt_divisibility — Discord as surface platform
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*66)
print("  SECTION 4: High-level check_dqt_divisibility — Discord surface")
print("="*66)

# Simulate Slack → Discord chain: observed DQT is now Discord's identity table.
# The high-level method should find NO ghost because Discord filters itself out.
ghost, err = bks.check_dqt_divisibility(discord_qt, "discord")
print(f"\n  surface=discord, observed≈discord_qt")
print(f"  ghost={ghost}, err={err}")
check(
    "High-level method returns (None, None) for Discord surface + Discord-like DQT",
    ghost is None and err is None,
    "Discord identity table erases all ghost traces — Forensic Horizon confirmed"
)

# Simulate Slack → Telegram chain: observed DQT is Telegram's. Ghost should be Slack.
ghost2, err2 = bks.check_dqt_divisibility(tg_qt, "telegram")
print(f"\n  surface=telegram, observed=Telegram Q-table (possible Slack ghost)")
print(f"  ghost={ghost2}, err={err2}")
check(
    "High-level method detects Slack ghost in Telegram-surface image",
    ghost2 == "slack",
    f"ghost={ghost2}, err={round(err2,4) if err2 else None}"
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Epsilon robustness: zeros in lib_dqt must not cause division errors
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*66)
print("  SECTION 5: Epsilon Robustness (zero-cell guard)")
print("="*66)

lib_with_zeros = slack_qt.copy()
lib_with_zeros[0] = 0  # inject a zero
lib_with_zeros[10] = 0

try:
    matched_z, meta_z = bks._check_pair(tg_qt, lib_with_zeros)
    check(
        "No division error with zero cells in lib_dqt (epsilon guard works)",
        True,
        f"matched={matched_z}, mean_error={meta_z['mean_error']:.4f}" if meta_z else f"matched={matched_z}"
    )
except Exception as e:
    check("No division error with zero cells in lib_dqt", False, f"Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*66)
total  = len(results)
passed = sum(1 for r in results if r[0] == "PASS")
failed = total - passed
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  — {failed} FAILED:")
    for tag, label in results:
        if tag == "FAIL":
            print(f"    ✗ {label}")
else:
    print("  — all assertions confirmed.")
print("="*66 + "\n")
