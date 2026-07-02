#!/usr/bin/env python3
"""
Generate supplementary figures using real Hetionet v1.0 data.

Figures produced (saved to supplementary/):
  - figS1_scatter.png: Concordance with continuous colormap
  - figS2_boxplot.png: Absolute log-scale error by degree class

Data: Real Hetionet v1.0 (47,031 nodes, 2,250,197 edges)
Metapath: CbG-GpPW (L=2), P=50
Runtime: ~2 minutes on Apple M2 Pro, 64 GB RAM.
"""

import sys
import os
import numpy as np
import time
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hetnetex.utils import load_hetionet, get_degrees
from hetnetex.xswap import xswap

# ── Configuration ──
P = 50
N_PAIRS = 30
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'supplementary')
CG = '#0F6E56'; CR = '#993556'


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    np.random.seed(42)

    # ── Load real Hetionet ──
    het = load_hetionet(download=True)
    A_cbg = het['A_cbg']; A_gpw = het['A_gpw']
    dc, dgc, mc = get_degrees(A_cbg)
    dgp, dp, mg = get_degrees(A_gpw)
    alpha = np.dot(dgc, dgp)

    print(f"\nCbG: {mc} edges, GpPW: {mg} edges, alpha = {alpha:.0f}")

    # ── Find nonzero L=2 pairs ──
    print("Finding nonzero pairs...")
    nz = []
    for s in range(het['n_compound']):
        if dc[s] == 0: continue
        row = A_cbg[s] * A_gpw
        for t in row.nonzero()[1]:
            val = row.toarray().flatten()[t]
            if val > 0:
                nz.append((s, t, val, dc[s] * dp[t]))
        if len(nz) >= 300: break

    nz.sort(key=lambda x: x[3])
    idx = np.linspace(0, len(nz) - 1, N_PAIRS).astype(int)
    sel = [nz[i] for i in idx]
    ps = np.array([p[0] for p in sel])
    pt = np.array([p[1] for p in sel])
    obs = np.array([p[2] for p in sel])
    dprod = np.array([p[3] for p in sel])
    hnx = np.array([dc[s] * dp[t] * alpha / (mc * mg) for s, t in zip(ps, pt)])

    # ── XSwap P=50 ──
    print(f"\nRunning XSwap P={P} on real Hetionet...")
    t0 = time.time()
    xs = np.zeros(N_PAIRS)
    for p in range(P):
        pA = xswap(A_cbg, multiplier=1)
        pB = xswap(A_gpw, multiplier=1)
        for k, (s, t) in enumerate(zip(ps, pt)):
            xs[k] += (pA[s] * pB)[:, t].toarray().sum()
        if (p + 1) % 25 == 0:
            print(f"  {p+1}/{P} ({time.time()-t0:.0f}s)")
    xs /= P

    v = (hnx > 0) & (xs > 0)
    rl = np.corrcoef(np.log10(hnx[v]+1), np.log10(xs[v]+1))[0, 1]
    rho = stats.spearmanr(hnx[v], xs[v])[0]
    print(f"\nResults: r_log = {rl:.4f}, rho = {rho:.4f}")

    med = np.median(dprod[v])
    low = dprod[v] <= med; high = dprod[v] > med
    hl = np.log10(hnx[v] + 1); xl = np.log10(xs[v] + 1)
    abs_err = np.abs(xl - hl)

    # ── Figure S1: Scatter with continuous colormap ──
    print("\nGenerating Figure S1...")
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(hl, xl, c=np.log10(dprod[v] + 1), cmap='RdYlGn_r',
                    alpha=0.7, s=50, edgecolors='none', zorder=3)
    lim = max(hl.max(), xl.max()) * 1.08
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.3)
    ax.set_xlabel('HetNetEX $\\log_{10}(\\mu+1)$', fontsize=11)
    ax.set_ylabel(f'XSwap $\\log_{{10}}(\\bar{{Y}}+1)$', fontsize=11)
    ax.set_title(f'Hetionet $L=2$ (CbG-GpPW, $n$=20,945, $P$={P})\n'
                 f'$r_{{log}}$={rl:.3f}, $\\rho$={rho:.3f}',
                 fontsize=11, fontweight='bold')
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label('$\\log_{10}(d_sd_t+1)$', fontsize=9)
    ax.grid(True, alpha=0.1); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'figS1_scatter.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figS1_scatter.png")

    # ── Figure S2: Box plot with absolute error ──
    print("Generating Figure S2...")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bp = ax.boxplot([abs_err[low], abs_err[high]], positions=[1, 2], widths=0.5,
                    patch_artist=True,
                    flierprops=dict(marker='o', markersize=5, alpha=0.4, markerfacecolor='gray'))
    bp['boxes'][0].set_facecolor(CG); bp['boxes'][0].set_alpha(0.65)
    bp['boxes'][1].set_facecolor(CR); bp['boxes'][1].set_alpha(0.65)
    for m_ in bp['medians']: m_.set_color('black'); m_.set_linewidth(2)
    for w_ in bp['whiskers']: w_.set_linewidth(1.5)
    for c_ in bp['caps']: c_.set_linewidth(1.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Low $d_sd_t$\n(specific nodes)', 'High $d_sd_t$\n(hub nodes)'], fontsize=11)
    ax.set_ylabel('Absolute error  $|\\log_{10}(\\bar{Y}+1) - \\log_{10}(\\mu+1)|$', fontsize=10)
    ax.set_title(f'Hetionet $L=2$, $P={P}$', fontsize=12, fontweight='bold')
    ax.set_ylim(bottom=0); ax.grid(True, alpha=0.15, axis='y')

    med_low = np.median(abs_err[low])
    med_high = np.median(abs_err[high])
    ax.text(1, med_low + 0.005, f'median = {med_low:.3f}',
            ha='center', fontsize=10, color=CG, fontweight='bold')
    ax.text(2, med_high + 0.005, f'median = {med_high:.3f}',
            ha='center', fontsize=10, color=CR, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'figS2_boxplot.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figS2_boxplot.png")

    print(f"\nDone. All supplementary figures saved to {OUTDIR}/")


if __name__ == '__main__':
    main()
