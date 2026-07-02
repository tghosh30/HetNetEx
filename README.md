# HetNetEX: Exact Asymptotic Inference for Heterogeneous Biomedical Knowledge Graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**HetNetEX** (**Het**erogeneous **Net**work **EX**act inference) computes exact asymptotic *p*-values for degree-weighted path counts (DWPC) in heterogeneous biomedical knowledge graphs (hetnets), replacing the permutation-based XSwap null model with closed-form configuration model theory.

## Key Results

| | XSwap (P=200) | HetNetEX |
|---|---|---|
| **Runtime (L=4)** | ~30,000 s | **0.05 s** |
| **p-value resolution** | ≥ 0.005 | **10⁻¹⁷** |
| **Speedup** | — | **10,000× – 1.3 billion×** |
| **Variance model** | Var ∝ μ² (gamma) | **(1+κ)μ** (correct) |
| **Concordance (ρ)** | reference | **> 0.96** |

## Installation

```bash
git clone https://github.com/tghosh30/HetNetEX.git
cd HetNetEX
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.9
- NumPy, SciPy, Matplotlib
- hetmatpy, hetnetpy (for real Hetionet validation)

## Quick Start

### Run HetNetEX on a pair

```python
from hetnetex.core import HetNetEX

# Load your hetnet adjacency matrices
model = HetNetEX(w=0.4)
model.add_edge_type('CbG', A_cbg, m_cbg)
model.add_edge_type('GpPW', A_gpw, m_gpw)

# Compute p-value for a source-target pair along a metapath
mu, kappa, sigma, p_value = model.compute(
    source=42, target=15,
    metapath=['CbG', 'GpPW'],
    observed_dwpc=0.110
)
```

### Reproduce paper figures

```bash
# Main text figures (synthetic network, P=200)
python scripts/generate_main_figures.py

# Supplementary figures (real Hetionet, P=50)
python scripts/generate_supplementary_figures.py

# Full Hetionet benchmark (requires ~2-6 hours)
python scripts/run_hetionet_benchmark.py --P 200 --metapaths all
```

## Repository Structure

```
HetNetEX/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── hetnetex/
│   ├── __init__.py
│   ├── core.py                        # HetNetEX algorithm (Algorithm 1)
│   ├── xswap.py                       # XSwap implementation for comparison
│   └── utils.py                       # Network construction, metrics
├── scripts/
│   ├── generate_main_figures.py       # Figures 1-3 from main text
│   ├── generate_supplementary_figures.py  # Figures S1-S2
│   └── run_hetionet_benchmark.py      # Full real Hetionet validation
├── figures/                           # Generated figures (PNG)
└── supplementary/                     # Supplementary figures (PNG)
```

## Method Overview

HetNetEX derives five theorems from the configuration model:

1. **Theorem 1 (Mean):** Expected null DWPC μ via rank-1 matrix chain in O(Ln)
2. **Theorem 2 (Variance):** Overdispersion κ from degree heterogeneity, giving Var = (1+κ)μ
3. **Theorem 3 (Poisson):** Distributional approximation for L ≤ 3 (μ small)
4. **Theorem 4 (CLT):** Normal approximation for L ≥ 4 (μ large)
5. **Theorem 5 (Equivalence):** HetNetEX = XSwap in the m → ∞ limit

Both methods target the same null model (degree-preserving random graphs). XSwap samples it with P permutations; HetNetEX computes it analytically.

## Reproducing Results

### Main Text (Synthetic Network, P=200)

```bash
python scripts/generate_main_figures.py
```

Generates:
- `figures/fig1_scatter_panel.png` — L=1–4 concordance (Fig. 1)
- `figures/fig2_boxplot_landscape.png` — Degree-stratified error + landscape (Fig. 2)
- `figures/fig3_Q1vsQ4.png` — Q1 vs Q4 at L=3 (Fig. 3)

Expected runtime: ~3 minutes on Apple M2 Pro.

### Supplementary (Real Hetionet, P=50)

```bash
python scripts/generate_supplementary_figures.py
```

Downloads Hetionet v1.0 automatically (16 MB), then generates:
- `supplementary/figS1_scatter.png` — Continuous colormap scatter
- `supplementary/figS2_boxplot.png` — Absolute error box plot

Expected runtime: ~2 minutes on Apple M2 Pro.

### Full Benchmark (Real Hetionet, P=200)

```bash
python scripts/run_hetionet_benchmark.py --P 200
```

Expected runtime: ~2–6 hours depending on metapaths tested.

## Citation

```bibtex
@inproceedings{ghosh2026hetnetex,
  title     = {HetNetEX: Exact Asymptotic Inference in Heterogeneous
               Biomedical Knowledge Graphs},
  author    = {Ghosh, Tusharkanti and Gillenwater, Lucas A. and
               Greene, Casey S. and Costello, James C.},
  booktitle = {Proceedings of IEEE BIBM},
  year      = {2026}
}
```

## Data Sources

- **Hetionet v1.0:** [https://het.io](https://het.io) — 47,031 nodes, 2,250,197 edges ([Himmelstein et al., 2017](https://doi.org/10.7554/eLife.26726))
- **XSwap:** [https://github.com/hetio/xswap](https://github.com/hetio/xswap) — ([Hanhijärvi et al., 2009](https://doi.org/10.1137/1.9781611972795.71))
- **Edge Prior:** ([Zietz et al., 2024](https://doi.org/10.1093/gigascience/giae001))

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Tusharkanti Ghosh — tusharkanti.ghosh@cuanschutz.edu
Department of Biostatistics and Informatics, Colorado School of Public Health
University of Colorado Anschutz Medical Campus
