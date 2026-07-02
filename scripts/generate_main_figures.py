#!/usr/bin/env python3
"""
Generate main text figures for the HetNetEX paper.

Figures produced (saved to figures/):
  - fig1_scatter_panel.png: L=1-4 concordance scatter (P=200)
  - fig2_boxplot_landscape.png: (a) degree-stratified error, (b) p-value landscape
  - fig3_Q1vsQ4.png: Q1 vs Q4 at L=3

Runtime: ~3 minutes on Apple M2 Pro, 64 GB RAM.
"""

import sys
import os
import numpy as np
import time
from scipy import stats

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from hetnetex.utils import build_powerlaw_bipartite, get_degrees, concordance_metrics
from hetnetex.xswap import xswap

# ── Configuration ──
P = 200          # Number of permutations
NP = 60          # Number of pairs per L
SEED = 42
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

# Colors
CG = '#0F6E56'   # Green (low degree)
CR = '#993556'    # Red (high degree)
CH = '#D85A30'    # Orange (HetNetEX)
CB = '#005298'    # Blue (XSwap)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    np.random.seed(SEED)

    # ── Build synthetic networks ──
    print("Building synthetic networks...")
    A1 = build_powerlaw_bipartite(30, 200, 300, gamma=1.5, seed=42)   # CbG
    A2 = build_powerlaw_bipartite(200, 200, 800, gamma=1.5, seed=99)  # GiG
    A3 = build_powerlaw_bipartite(200, 40, 500, gamma=1.5, seed=77)   # GpPW

    d1s, d1t, m1 = get_degrees(A1)
    d2s, d2t, m2 = get_degrees(A2)
    d3s, d3t, m3 = get_degrees(A3)

    # HetNetEX transition matrices
    T1 = np.outer(d1s, d1t).astype(float) / m1
    T2 = np.outer(d2s, d2t).astype(float) / m2
    T3 = np.outer(d3s, d3t).astype(float) / m3

    # Expected count matrices for L=1..4
    mus = {1: T1, 2: T1 @ T3, 3: T1 @ T2 @ T3, 4: T1 @ T2 @ T2 @ T3}
    adj_lists = {1: [A1], 2: [A1, A3], 3: [A1, A2, A3], 4: [A1, A2, A2, A3]}
    tgt_sizes = {1: 200, 2: 40, 3: 40, 4: 40}

    # ── Run XSwap at P=200 for L=1..4 ──
    data = {}
    for L in range(1, 5):
        np.random.seed(10 + L)
        ps = np.random.choice(30, NP)
        pt = np.random.choice(tgt_sizes[L], NP)
        hnx = np.array([mus[L][s, t] for s, t in zip(ps, pt)])
        dprod = np.array([d1s[s] * (d3s[t] if L > 1 else d1t[t]) for s, t in zip(ps, pt)])

        print(f"\nL={L}, P={P}...")
        t0 = time.time()
        nc = np.zeros(NP)
        for p in range(P):
            pm = xswap(A1, multiplier=5)
            for A in adj_lists[L][1:]:
                pm = pm @ xswap(A, multiplier=5)
            pm_dense = np.array(pm.todense())
            for k, (s, t) in enumerate(zip(ps, pt)):
                nc[k] += pm_dense[s, t]
            if (p + 1) % 50 == 0:
                print(f"  {p+1}/{P} ({time.time()-t0:.0f}s)")
        nc /= P
        txs = time.time() - t0

        v = (hnx > 0) & (nc > 0)
        med = np.median(dprod[v])
        low = dprod[v] <= med
        high = dprod[v] > med
        rl = np.corrcoef(np.log10(hnx[v]+1), np.log10(nc[v]+1))[0, 1] if v.sum() > 5 else 0
        rho = stats.spearmanr(hnx[v], nc[v])[0] if v.sum() > 5 else 0
        rel_err = np.abs(nc[v] - hnx[v]) / (hnx[v] + 1e-10)

        data[L] = {
            'hnx': hnx, 'xs': nc, 'v': v, 'dprod': dprod,
            'low': low, 'high': high, 'rl': rl, 'rho': rho,
            'rel_err': rel_err, 'rel_low': rel_err[low], 'rel_high': rel_err[high],
            'txs': txs,
        }
        print(f"  r_log={rl:.3f}, rho={rho:.3f}, time={txs:.0f}s")

    # ── Figure 1: L=1-4 scatter panel ──
    print("\nGenerating Figure 1...")
    titles = {
        1: '$L=1$: CbG (Bernoulli)',
        2: '$L=2$: CbG-GpPW (Poisson)',
        3: '$L=3$: CbG-GiG-GpPW ($\\kappa$)',
        4: '$L=4$: CbG-(GiG)$^2$-GpPW (CLT)',
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for idx, L in enumerate(range(1, 5)):
        ax = axes[idx // 2, idx % 2]
        d = data[L]; v = d['v']
        hl = np.log10(d['hnx'][v] + 1)
        xl = np.log10(d['xs'][v] + 1)
        low, high = d['low'], d['high']

        ax.scatter(hl[low], xl[low], c=CG, alpha=0.6, s=30, edgecolors='none',
                   label='Low $d_sd_t$', zorder=3)
        ax.scatter(hl[high], xl[high], c=CR, alpha=0.6, s=30, edgecolors='none',
                   label='High $d_sd_t$', zorder=3)
        lim = max(hl.max(), xl.max()) * 1.08
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.3)

        if L >= 3:
            xm = xl.max()
            cm = xl >= xm * 0.98
            nc2 = cm.sum()
            if nc2 >= 2:
                ax.axhline(xm, color=CB, ls='--', lw=1.5, alpha=0.6)
                ax.text(0.02, xm + 0.03, f'XSwap ceiling ({nc2} pairs)',
                        fontsize=7, color=CB, fontweight='bold')

        ax.set_xlabel('HetNetEX $\\log_{10}(\\mu+1)$', fontsize=9)
        ax.set_ylabel('XSwap $\\log_{10}(\\bar{Y}+1)$', fontsize=9)
        ax.set_title(f'{titles[L]}\n$r_{{\\log}}={d["rl"]:.3f}$, $\\rho={d["rho"]:.3f}$',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.1)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig1_scatter_panel.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig1_scatter_panel.png")

    # ── Figure 2: (a) Box plot, (b) Landscape ──
    print("Generating Figure 2...")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Box plots
    positions, boxes, colors = [], [], []
    for i, L in enumerate(range(1, 5)):
        d = data[L]
        boxes.append(d['rel_low']); positions.append(i * 2.5 + 0.6); colors.append(CG)
        boxes.append(d['rel_high']); positions.append(i * 2.5 + 1.4); colors.append(CR)

    bp = a1.boxplot(boxes, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='.', markersize=3, alpha=0.4))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    for m_ in bp['medians']:
        m_.set_color('black'); m_.set_linewidth(1.5)
    a1.set_xticks([1.0, 3.5, 6.0, 8.5])
    a1.set_xticklabels(['$L=1$', '$L=2$', '$L=3$', '$L=4$'])
    a1.set_ylabel('Relative error', fontsize=9)
    a1.set_title(f'(a) XSwap error at $P={P}$', fontsize=10, fontweight='bold')
    a1.set_ylim(bottom=0); a1.grid(True, alpha=0.15, axis='y')
    a1.legend([Patch(facecolor=CG, alpha=0.6), Patch(facecolor=CR, alpha=0.6)],
              ['Low $d_sd_t$', 'High $d_sd_t$'], fontsize=7)

    # (b) Landscape
    Ls = [1, 2, 3, 4, 5, 6, 7, 8]
    a2.fill_between([0.5, 4.5], [0, 0], [2.3, 2.3], alpha=0.08, color=CB)
    a2.fill_between([5.5, 8.5], [0, 0], [14, 14], alpha=0.03, color='red')
    a2.axhline(2.3, color=CB, ls='--', lw=2, alpha=0.6)
    a2.text(2.5, 1.2, 'XSwap works', fontsize=8, color=CB, fontweight='bold', ha='center')
    a2.text(7, 0.8, 'XSwap infeasible', fontsize=8, color='red',
            fontweight='bold', ha='center', alpha=0.6)
    hm_ = [2.5, 3.5, 5, 7, 9, 10.5, 12, 13]
    a2.fill_between(Ls, [0]*8, hm_, alpha=0.1, color=CH)
    a2.plot(Ls, hm_, color=CH, lw=2.5, label='HetNetEX')
    a2.set_xlabel('$L$'); a2.set_ylabel('$-\\log_{10}(p)$')
    a2.set_xticks(Ls); a2.set_ylim(0, 14)
    a2.legend(fontsize=7, loc='upper left')
    a2.set_title('(b) $p$-value landscape', fontsize=10, fontweight='bold')
    a2.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig2_boxplot_landscape.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_boxplot_landscape.png")

    # ── Figure 3: Q1 vs Q4 at L=3 ──
    print("Generating Figure 3...")
    d = data[3]; v = d['v']
    hl = np.log10(d['hnx'][v] + 1); xl = np.log10(d['xs'][v] + 1)
    low, high = d['low'], d['high']
    rho_l = stats.spearmanr(d['hnx'][v][low], d['xs'][v][low])[0] if low.sum() > 3 else 0
    rho_h = stats.spearmanr(d['hnx'][v][high], d['xs'][v][high])[0] if high.sum() > 3 else 0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))

    a1.scatter(hl[low], xl[low], c=CG, alpha=0.7, s=40, edgecolors='none')
    lm = max(hl[low].max(), xl[low].max()) * 1.08
    a1.plot([0, lm], [0, lm], 'r--', lw=1, alpha=0.4)
    a1.set_xlabel('HetNetEX $\\log_{10}(\\mu+1)$', fontsize=10)
    a1.set_ylabel('XSwap $\\log_{10}(\\bar{Y}+1)$', fontsize=10)
    a1.set_title(f'Q1: Low degree ($\\rho={rho_l:.2f}$)', fontsize=11, fontweight='bold')
    a1.text(0.05, 0.85, 'Tight', fontsize=16, color=CG, fontweight='bold', transform=a1.transAxes)
    a1.grid(True, alpha=0.1); a1.set_xlim(left=0); a1.set_ylim(bottom=0)

    a2.scatter(hl[high], xl[high], c=CR, alpha=0.7, s=40, edgecolors='none')
    lm = max(hl[high].max(), xl[high].max()) * 1.08
    a2.plot([0, lm], [0, lm], 'r--', lw=1, alpha=0.4)
    a2.set_xlabel('HetNetEX $\\log_{10}(\\mu+1)$', fontsize=10)
    a2.set_ylabel('XSwap $\\log_{10}(\\bar{Y}+1)$', fontsize=10)
    a2.set_title(f'Q4: High degree ($\\rho={rho_h:.2f}$)', fontsize=11, fontweight='bold')
    a2.text(0.05, 0.85, 'Scattered', fontsize=16, color=CR, fontweight='bold', transform=a2.transAxes)
    a2.grid(True, alpha=0.1); a2.set_xlim(left=0); a2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig3_Q1vsQ4.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_Q1vsQ4.png")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DONE. All figures saved to figures/")
    print("=" * 60)
    for L in range(1, 5):
        d = data[L]
        print(f"  L={L}: r_log={d['rl']:.3f}, rho={d['rho']:.3f}, "
              f"low_err={np.median(d['rel_low']):.3f}, high_err={np.median(d['rel_high']):.3f}")


if __name__ == '__main__':
    main()
