"""
Utility functions for network construction, data loading, and evaluation metrics.
"""

import numpy as np
from scipy import sparse, stats


def build_powerlaw_bipartite(n_source, n_target, m, gamma=1.5, seed=42):
    """
    Build a random bipartite graph with power-law degree distribution.

    Parameters
    ----------
    n_source, n_target : int
        Number of source and target nodes.
    m : int
        Number of edges.
    gamma : float, default=1.5
        Power-law exponent. Lower = more skewed (more hubs).
    seed : int
        Random seed.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Adjacency matrix (n_source x n_target).
    """
    rng = np.random.RandomState(seed)
    source_probs = np.arange(1, n_source + 1, dtype=float) ** (-gamma)
    source_probs /= source_probs.sum()
    target_probs = np.arange(1, n_target + 1, dtype=float) ** (-gamma)
    target_probs /= target_probs.sum()

    edges = set()
    max_edges = min(m, n_source * n_target - 1)
    while len(edges) < max_edges:
        s = rng.choice(n_source, p=source_probs)
        t = rng.choice(n_target, p=target_probs)
        edges.add((s, t))

    rows, cols = zip(*edges)
    A = sparse.csr_matrix(
        (np.ones(len(edges)), (list(rows), list(cols))),
        shape=(n_source, n_target)
    )
    return A


def get_degrees(A):
    """Return (source_degrees, target_degrees, n_edges) from adjacency."""
    d_source = np.array(A.sum(axis=1)).flatten()
    d_target = np.array(A.sum(axis=0)).flatten()
    return d_source, d_target, A.nnz


def concordance_metrics(x, y):
    """
    Compute concordance metrics between two arrays.

    Returns dict with: pearson_raw, pearson_log, spearman, n_valid
    """
    valid = (x > 0) & (y > 0)
    n_valid = valid.sum()

    if n_valid < 3:
        return {'pearson_raw': np.nan, 'pearson_log': np.nan,
                'spearman': np.nan, 'n_valid': n_valid}

    r_raw = np.corrcoef(x[valid], y[valid])[0, 1]
    r_log = np.corrcoef(np.log10(x[valid] + 1), np.log10(y[valid] + 1))[0, 1]
    rho, _ = stats.spearmanr(x[valid], y[valid])

    return {
        'pearson_raw': r_raw,
        'pearson_log': r_log,
        'spearman': rho,
        'n_valid': n_valid,
    }


def degree_stratified_error(hnx, xs, dprod, metric='relative'):
    """
    Compute error stratified by degree product.

    Parameters
    ----------
    hnx : array
        HetNetEX expected counts.
    xs : array
        XSwap empirical means.
    dprod : array
        Degree products (d_s * d_t).
    metric : str
        'relative' for |xs-hnx|/hnx, 'absolute_log' for |log(xs+1)-log(hnx+1)|

    Returns
    -------
    dict with low_error, high_error, low_mask, high_mask
    """
    valid = (hnx > 0) & (xs > 0)
    median_dp = np.median(dprod[valid])
    low = dprod[valid] <= median_dp
    high = dprod[valid] > median_dp

    if metric == 'relative':
        err = np.abs(xs[valid] - hnx[valid]) / (hnx[valid] + 1e-10)
    else:
        err = np.abs(np.log10(xs[valid] + 1) - np.log10(hnx[valid] + 1))

    return {
        'error': err,
        'low_mask': low,
        'high_mask': high,
        'low_error': err[low],
        'high_error': err[high],
        'low_median': np.median(err[low]) if low.sum() > 0 else np.nan,
        'high_median': np.median(err[high]) if high.sum() > 0 else np.nan,
    }


def load_hetionet(filepath='hetionet-v1.0.json.bz2', download=True):
    """
    Load Hetionet v1.0 and build adjacency matrices.

    Parameters
    ----------
    filepath : str
        Path to hetionet-v1.0.json.bz2
    download : bool
        If True and file not found, download from GitHub.

    Returns
    -------
    dict with adjacency matrices and degree vectors for CbG, GiG, GpPW
    """
    import os
    import json
    import bz2
    from collections import defaultdict

    if not os.path.exists(filepath) and download:
        import urllib.request
        url = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/hetnet.json.bz2'
        print(f"Downloading Hetionet v1.0 from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded: {os.path.getsize(filepath) / 1e6:.1f} MB")

    print("Loading Hetionet v1.0...")
    with bz2.open(filepath, 'rt') as f:
        het = json.load(f)

    # Build node ID maps
    node_map = {}
    kind_nodes = defaultdict(list)
    for n in het['nodes']:
        kind = n['kind']
        ident = n['identifier']
        node_map[(kind, ident)] = len(kind_nodes[kind])
        kind_nodes[kind].append(n)

    nc = len(kind_nodes['Compound'])
    ng = len(kind_nodes['Gene'])
    npw = len(kind_nodes['Pathway'])

    print(f"  Compounds: {nc}, Genes: {ng}, Pathways: {npw}")

    # Build CbG (Compound-binds-Gene)
    cbg_r, cbg_c = [], []
    for e in het['edges']:
        if e['kind'] == 'binds':
            si = node_map.get(('Compound', e['source_id'][1]))
            ti = node_map.get(('Gene', e['target_id'][1]))
            if si is None:
                si = node_map.get(('Compound', e['target_id'][1]))
                ti = node_map.get(('Gene', e['source_id'][1]))
            if si is not None and ti is not None:
                cbg_r.append(si)
                cbg_c.append(ti)
    A_cbg = sparse.csr_matrix(
        (np.ones(len(cbg_r)), (cbg_r, cbg_c)), shape=(nc, ng))

    # Build GiG (Gene-interacts-Gene)
    gig_r, gig_c = [], []
    for e in het['edges']:
        if e['kind'] == 'interacts':
            si = node_map.get(('Gene', e['source_id'][1]))
            ti = node_map.get(('Gene', e['target_id'][1]))
            if si is not None and ti is not None:
                gig_r.extend([si, ti])
                gig_c.extend([ti, si])
    A_gig = sparse.csr_matrix(
        (np.ones(len(gig_r)), (gig_r, gig_c)), shape=(ng, ng))

    # Build GpPW (Gene-participates-Pathway)
    gpw_r, gpw_c = [], []
    for e in het['edges']:
        if e['kind'] == 'participates':
            si = node_map.get(('Gene', e['source_id'][1]))
            ti = node_map.get(('Pathway', e['target_id'][1]))
            if si is None:
                si = node_map.get(('Gene', e['target_id'][1]))
                ti = node_map.get(('Pathway', e['source_id'][1]))
            if si is not None and ti is not None:
                gpw_r.append(si)
                gpw_c.append(ti)
    A_gpw = sparse.csr_matrix(
        (np.ones(len(gpw_r)), (gpw_r, gpw_c)), shape=(ng, npw))

    print(f"  CbG: {A_cbg.shape}, m={A_cbg.nnz}")
    print(f"  GiG: {A_gig.shape}, m={A_gig.nnz // 2} (undirected)")
    print(f"  GpPW: {A_gpw.shape}, m={A_gpw.nnz}")

    return {
        'A_cbg': A_cbg, 'A_gig': A_gig, 'A_gpw': A_gpw,
        'n_compound': nc, 'n_gene': ng, 'n_pathway': npw,
        'kind_nodes': dict(kind_nodes), 'node_map': dict(node_map),
    }
