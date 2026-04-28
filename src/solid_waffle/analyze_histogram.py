"""
analyze_histogram.py

Usage:
    python analyze_histogram.py output_hist.txt

What this does:
    1. Loads the histogram saved by histograms.py
    2. Plots the raw histogram (all active channels overlaid)
    3. Smooths each channel with a Gaussian (sigma=128 bins)
    4. Computes hist / smoothed_hist - 1 for each channel
    5. Plots the ratio, which reveals anomalous ADU values
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ── 0. Load ──────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python analyze_histogram.py <output_hist.txt>")
    sys.exit()

histfile = sys.argv[1]
hist = np.loadtxt(histfile, dtype=int)   # shape: (65536, 32)

# ADU axis: just the bin centres 0, 1, 2, ... 65535
adu = np.arange(hist.shape[0])

# Only use channels that actually have data
active_channels = [ch for ch in range(32) if hist[:, ch].max() > 0]
print(f"Active channels: {active_channels}")

# Grid layout used by all three figures
ncols = 4
nrows = int(np.ceil(len(active_channels) / ncols))

# ── 1. Gaussian smoothing ─────────────────────────────────────────────────────
# gaussian_filter1d slides a Gaussian kernel along the 1-D array.
# sigma=128 means the kernel is ~128 bins wide (one "blur radius").
# A larger sigma = more smoothing = broader features survive.
# truncate=4.0 means the kernel extends 4*sigma bins either side before
# being cut off — the default; included here so it's visible.

sigma = 128
smoothed = np.zeros_like(hist, dtype=float)

for ch in active_channels:
    smoothed[:, ch] = gaussian_filter1d(hist[:, ch].astype(float),
                                        sigma=sigma, truncate=4.0)

# ── 2. Ratio: hist / smoothed - 1 ────────────────────────────────────────────
# Where hist == smoothed the ratio is 0 (flat baseline).
# Positive spikes  →  more counts than expected at that ADU value.
# Negative dips    →  fewer counts than expected.
# np.errstate suppresses divide-by-zero warnings for empty bins;
# those bins become NaN and are simply not plotted.

with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(smoothed > 0, hist / smoothed - 1.0, np.nan)

# ── 3. Figure 1: raw histogram (one panel per channel) ───────────────────────

ncols = 4
nrows = int(np.ceil(len(active_channels) / ncols))

fig1, axes1 = plt.subplots(nrows, ncols,
                            figsize=(ncols * 4, nrows * 3),
                            sharex=True)
axes1 = np.array(axes1).flatten()

for idx, ch in enumerate(active_channels):
    ax = axes1[idx]
    ax.semilogy(adu, hist[:, ch], lw=0.5, color="steelblue")
    ax.set_title(f"Ch {ch}", fontsize=9)
    ax.set_xlim(0, 65535)

for idx in range(len(active_channels), len(axes1)):
    axes1[idx].set_visible(False)

fig1.supxlabel("ADU value (pixel intensity)")
fig1.supylabel("Counts (log scale)")
fig1.suptitle("Raw histogram — all active channels")
plt.tight_layout()
plt.savefig("histogram_raw.png", dpi=150)
print("Saved histogram_raw.png")

# ── 4. Figure 2: smoothed histogram overlaid on raw (one panel per channel) ──
# This lets you check that the smoothing looks sensible before trusting
# the ratio plot.

fig2, axes2 = plt.subplots(nrows, ncols,
                            figsize=(ncols * 4, nrows * 3),
                            sharex=True)
axes2 = np.array(axes2).flatten()

for idx, ch in enumerate(active_channels):
    ax = axes2[idx]
    ax.semilogy(adu, hist[:, ch], lw=0.5, color="steelblue", label="raw")
    ax.semilogy(adu, smoothed[:, ch], lw=1.2, color="orange", label="smoothed")
    ax.set_title(f"Ch {ch}", fontsize=9)
    ax.set_xlim(0, 65535)
    if idx == 0:
        ax.legend(fontsize=7)

# Hide any unused subplot panels
for idx in range(len(active_channels), len(axes2)):
    axes2[idx].set_visible(False)

fig2.supxlabel("ADU value")
fig2.supylabel("Counts (log scale)")
fig2.suptitle(f"Raw vs Gaussian-smoothed histogram (σ={sigma} bins)")
plt.tight_layout()
plt.savefig("histogram_smoothed_overlay.png", dpi=150)
print("Saved histogram_smoothed_overlay.png")

# ── 5. Figure 3: ratio hist/smoothed - 1 ─────────────────────────────────────
# Zoom into a narrower ADU range here — most interesting features are
# not at the very edges. Adjust xlim to suit your data.

fig3, axes3 = plt.subplots(nrows, ncols,
                            figsize=(ncols * 4, nrows * 3),
                            sharex=True, sharey=True)
axes3 = np.array(axes3).flatten()

for idx, ch in enumerate(active_channels):
    ax = axes3[idx]
    ax.plot(adu, ratio[:, ch], lw=0.5, color="crimson")
    ax.axhline(0, color="black", lw=0.8, ls="--")   # zero line for reference
    ax.set_title(f"Ch {ch}", fontsize=9)
    ax.set_xlim(0, 65535)
    # Clip the y-axis so big spikes at the edges don't crush the middle detail
    ax.set_ylim(-0.5, 0.5)

for idx in range(len(active_channels), len(axes3)):
    axes3[idx].set_visible(False)

fig3.supxlabel("ADU value")
fig3.supylabel("hist / smoothed  –  1")
fig3.suptitle(f"Ratio plot (σ={sigma}): deviations from smooth background")
plt.tight_layout()
plt.savefig("histogram_ratio.png", dpi=150)
print("Saved histogram_ratio.png")

plt.show()