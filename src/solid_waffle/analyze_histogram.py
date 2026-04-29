"""
analyze_histogram.py

Usage:
    python analyze_histogram.py output_hist.txt

Produces 6 output figures:
    histogram_raw_combined.png          — all channels overlaid, one panel
    histogram_raw_grid.png              — one panel per channel
    histogram_smoothed_combined.png     — raw vs smoothed overlaid, one panel
    histogram_smoothed_grid.png         — raw vs smoothed, one panel per channel
    histogram_ratio_combined.png        — fractional residual overlaid, one panel
    histogram_ratio_grid.png            — fractional residual, one panel per channel
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

adu = np.arange(hist.shape[0])           # ADU axis: 0, 1, 2, ... 65535

active_channels = [ch for ch in range(32) if hist[:, ch].max() > 0]
print(f"Active channels: {active_channels}")

# ── 1. Gaussian smoothing ─────────────────────────────────────────────────────
sigma = 1024
smoothed = np.zeros_like(hist, dtype=float)

for ch in active_channels:
    smoothed[:, ch] = gaussian_filter1d(hist[:, ch].astype(float),
                                        sigma=sigma, truncate=4.0)

# ── 2. Ratio: hist / smoothed - 1 ────────────────────────────────────────────

with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(smoothed > 0, hist / smoothed - 1.0, np.nan)

# ── Helper: make the per-channel grid figure ──────────────────────────────────

def make_grid_fig(active_channels, title):
    """Create a grid of subplots, one per channel. Returns fig and flat axes array."""
    ncols = 4
    nrows = int(np.ceil(len(active_channels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    fig.suptitle(title)
    # Hide unused panels (e.g. if 16 channels only fills 4 rows exactly, no extras)
    for idx in range(len(active_channels), len(axes)):
        axes[idx].set_visible(False)
    return fig, axes

# ── 3. Raw histogram ──────────────────────────────────────────────────────────

# -- Combined: all channels overlaid in one panel
fig, ax = plt.subplots(figsize=(12, 5))
for ch in active_channels:
    ax.semilogy(adu, hist[:, ch], lw=0.5, alpha=0.6, label=f"Ch {ch}")
ax.set_xlabel("ADU value (pixel intensity)")
ax.set_ylabel("Counts (log scale)")
ax.set_title("Raw histogram — all active channels")
ax.set_xlim(0, 65535)
ax.legend(fontsize=6, ncol=4, loc="upper right")
plt.tight_layout()
plt.savefig("histogram_raw_combined.png", dpi=150)
print("Saved histogram_raw_combined.png")
plt.close()

# -- Grid: one panel per channel
fig, axes = make_grid_fig(active_channels, "Raw histogram — per channel")
for idx, ch in enumerate(active_channels):
    axes[idx].semilogy(adu, hist[:, ch], lw=0.5, color="steelblue")
    axes[idx].set_title(f"Ch {ch}", fontsize=9)
    axes[idx].set_xlim(0, 65535)
fig.supxlabel("ADU value (pixel intensity)")
fig.supylabel("Counts (log scale)")
plt.tight_layout()
plt.savefig("histogram_raw_grid.png", dpi=150)
print("Saved histogram_raw_grid.png")
plt.close()

# ── 4. Raw vs smoothed ────────────────────────────────────────────────────────

# -- Combined
fig, ax = plt.subplots(figsize=(12, 5))
for ch in active_channels:
    ax.semilogy(adu, hist[:, ch], lw=0.5, alpha=0.4, color="steelblue",
                label="raw" if ch == active_channels[0] else "_nolegend_")
    ax.semilogy(adu, smoothed[:, ch], lw=1.0, alpha=0.8, color="orange",
                label="smoothed" if ch == active_channels[0] else "_nolegend_")
ax.set_xlabel("ADU value (pixel intensity)")
ax.set_ylabel("Counts (log scale)")
ax.set_title(f"Raw vs smoothed (σ={sigma} bins) — all channels")
ax.set_xlim(0, 65535)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("histogram_smoothed_combined.png", dpi=150)
print("Saved histogram_smoothed_combined.png")
plt.close()

# -- Grid
fig, axes = make_grid_fig(active_channels, f"Raw vs smoothed (σ={sigma}) — per channel")
for idx, ch in enumerate(active_channels):
    axes[idx].semilogy(adu, hist[:, ch], lw=0.5, color="steelblue",
                       label="raw" if idx == 0 else "_nolegend_")
    axes[idx].semilogy(adu, smoothed[:, ch], lw=1.0, color="orange",
                       label="smoothed" if idx == 0 else "_nolegend_")
    axes[idx].set_title(f"Ch {ch}", fontsize=9)
    axes[idx].set_xlim(0, 65535)
axes[0].legend(fontsize=7)
fig.supxlabel("ADU value (pixel intensity)")
fig.supylabel("Counts (log scale)")
plt.tight_layout()
plt.savefig("histogram_smoothed_grid.png", dpi=150)
print("Saved histogram_smoothed_grid.png")
plt.close()

# ── 5. Fractional residual ────────────────────────────────────────────────────

# -- Combined
fig, ax = plt.subplots(figsize=(12, 5))
for ch in active_channels:
    ax.plot(adu, ratio[:, ch], lw=0.5, alpha=0.6, label=f"Ch {ch}")
ax.axhline(0, color="black", lw=1.0, ls="--")
ax.set_xlabel("ADU value (pixel intensity)")
ax.set_ylabel("hist / smoothed  –  1")
ax.set_title(f"Fractional residual (σ={sigma}) — all channels")
ax.set_xlim(0, 65535)
ax.set_ylim(-5, 5)
ax.legend(fontsize=6, ncol=4, loc="upper right")
plt.tight_layout()
plt.savefig("histogram_ratio_combined.png", dpi=150)
print("Saved histogram_ratio_combined.png")
plt.close()

# -- Grid
fig, axes = make_grid_fig(active_channels, f"Fractional residual (σ={sigma}) — per channel")
for idx, ch in enumerate(active_channels):
    axes[idx].plot(adu, ratio[:, ch], lw=0.5, color="crimson")
    axes[idx].axhline(0, color="black", lw=0.8, ls="--")
    axes[idx].set_title(f"Ch {ch}", fontsize=9)
    axes[idx].set_xlim(0, 65535)
    axes[idx].set_ylim(-5, 5)
fig.supxlabel("ADU value (pixel intensity)")
fig.supylabel("hist / smoothed  –  1")
plt.tight_layout()
plt.savefig("histogram_ratio_grid.png", dpi=150)
print("Saved histogram_ratio_grid.png")
plt.close()

print("Done! All 6 figures saved.")