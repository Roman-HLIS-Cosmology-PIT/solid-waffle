"""
analyze_histogram.py
 
Usage:
    python analyze_histogram.py output_hist.txt
 
What this does:
    1. Loads the histogram saved by histograms.py
    2. Plots the raw histogram (all active channels overlaid)
    3. Smooths each channel with a Gaussian (sigma=128 bins)
    4. Computes hist / smoothed_hist - 1 for each channel
    5. Plots the ratio, which reveals sticky ADU values
"""
 
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
 
# Load histogram
 
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

# Gaussian smoothing
sigma = 128
smoothed = np.zeros_like(hist, dtype=float)
 
for ch in active_channels:
    smoothed[:, ch] = gaussian_filter1d(hist[:, ch].astype(float),
                                        sigma=sigma, truncate=4.0)
 
# Find hist / smoothed - 1 
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(smoothed > 0, hist / smoothed - 1.0, np.nan)
 
# output raw histogram
fig1, ax1 = plt.subplots(figsize=(12, 5))
 
for ch in active_channels:
    ax1.semilogy(adu, hist[:, ch], lw=0.5, alpha=0.6, label=f"Ch {ch}")
 
ax1.set_xlabel("ADU value (pixel intensity)")
ax1.set_ylabel("Counts (log scale)")
ax1.set_title("Raw histogram — all active channels")
ax1.set_xlim(0, 65535)
# Put the legend outside the plot so it doesn't cover the data
ax1.legend(fontsize=6, ncol=4, loc="upper right",
           bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.savefig("histogram_raw.png", dpi=150)
print("Saved histogram_raw.png")
 
# output smoothed and normal histogram
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
 
# output residuals graph
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