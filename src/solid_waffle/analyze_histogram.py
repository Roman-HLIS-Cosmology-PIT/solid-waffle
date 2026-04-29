"""
analyze_histogram.py

Usage:
    python analyze_histogram.py output_hist.txt

What this does:
    1. Loads the histogram saved by histograms.py
    2. Plots the raw histogram — all channels overlaid in one figure
    3. Smooths each channel with a Gaussian (sigma=128 bins)
    4. Plots raw vs smoothed — all channels overlaid in one figure
    5. Computes hist / smoothed - 1 and plots all channels overlaid in one figure
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ── 0. Load ───────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python analyze_histogram.py <output_hist.txt>")
    sys.exit()

histfile = sys.argv[1]
hist = np.loadtxt(histfile, dtype=int)   # shape: (65536, 32)

# ADU axis: bin centres 0, 1, 2, ... 65535
adu = np.arange(hist.shape[0])

# Only use channels that actually have data
all_channels = [ch for ch in range(32) if hist[:, ch].max() >= 0]
print(f"Active channels: {active_channels}")

# ── 1. Gaussian smoothing ─────────────────────────────────────────────────────
# gaussian_filter1d slides a Gaussian kernel along the 1-D histogram.
# sigma=128 means the blur radius is 128 bins wide.
# Larger sigma = smoother baseline, fewer fine features survive.

sigma = 512
smoothed = np.zeros_like(hist, dtype=float)

for ch in all_channels:
    smoothed[:, ch] = gaussian_filter1d(hist[:, ch].astype(float),
                                        sigma=sigma, truncate=4.0)

# ── 2. Ratio: hist / smoothed - 1 ────────────────────────────────────────────
# Result is 0 where data matches the smooth baseline.
# Positive spikes = more counts than expected at that ADU value.
# Negative dips   = fewer counts than expected.
# Bins where smoothed==0 are set to NaN so they are silently skipped.

with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(smoothed > 0, hist / smoothed - 1.0, np.nan)

# ── 3. Figure 1: raw histogram, all channels overlaid ────────────────────────

fig1, ax1 = plt.subplots(figsize=(12, 5))

for ch in active_channels:
    ax1.semilogy(adu, hist[:, ch], lw=0.5, alpha=0.6, label=f"Ch {ch}")

ax1.set_xlabel("ADU value (pixel intensity)")
ax1.set_ylabel("Counts (log scale)")
ax1.set_title("Raw histogram — all active channels")
ax1.set_xlim(0, 65535)
ax1.legend(fontsize=6, ncol=4, loc="upper right")
plt.tight_layout()
plt.savefig("histogram_raw.png", dpi=150)
print("Saved histogram_raw.png")

# ── 4. Figure 2: raw vs smoothed, all channels overlaid ──────────────────────
# Raw channels are plotted in blue, smoothed in orange.
# The smoothed lines should sit cleanly on top of the raw data
# if sigma is a good choice.

fig2, ax2 = plt.subplots(figsize=(12, 5))

for ch in active_channels:
    ax2.semilogy(adu, hist[:, ch], lw=0.5, alpha=0.4, color="steelblue",
                 label="raw" if ch == active_channels[0] else "_nolegend_")
    ax2.semilogy(adu, smoothed[:, ch], lw=1.0, alpha=0.8, color="orange",
                 label="smoothed" if ch == active_channels[0] else "_nolegend_")

ax2.set_xlabel("ADU value (pixel intensity)")
ax2.set_ylabel("Counts (log scale)")
ax2.set_title(f"Raw vs Gaussian-smoothed histogram (σ={sigma} bins)")
ax2.set_xlim(0, 65535)
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("histogram_smoothed_overlay.png", dpi=150)
print("Saved histogram_smoothed_overlay.png")

# ── 5. Figure 3: fractional residual, all channels overlaid ──────────────────
# This is the main science result.
# ylim is clipped to ±0.5 — widen if your features are larger than ±50%.

fig3, ax3 = plt.subplots(figsize=(12, 5))

for ch in active_channels:
    ax3.plot(adu, ratio[:, ch], lw=0.5, alpha=0.6, label=f"Ch {ch}")

ax3.axhline(0, color="black", lw=1.0, ls="--", label="zero")
ax3.set_xlabel("ADU value (pixel intensity)")
ax3.set_ylabel("hist / smoothed  –  1")
ax3.set_title(f"Fractional residual (σ={sigma} bins)")
ax3.set_xlim(0, 65535)
ax3.set_ylim(-5, 5)
ax3.legend(fontsize=6, ncol=4, loc="upper right")
plt.tight_layout()
plt.savefig("histogram_ratio.png", dpi=150)
print("Saved histogram_ratio.png")

plt.show()