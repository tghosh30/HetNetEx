"""
XSwap edge-swapping implementation for generating degree-preserving null graphs.

Reference: Hanhijärvi et al., "Randomization techniques for graphs," SIAM SDM, 2009.
Adapted for hetnets: Himmelstein & Baranzini, PLoS Comput. Biol., 2015.
"""

import numpy as np
from scipy import sparse


def xswap(adjacency, multiplier=10, seed=None):
    """
    Generate a degree-preserving random graph via edge swapping.

    Parameters
    ----------
    adjacency : scipy.sparse matrix
        Adjacency matrix (bipartite or square).
    multiplier : int, default=10
        Number of swap attempts = multiplier * number_of_edges.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    permuted : scipy.sparse.csr_matrix
        Permuted adjacency matrix with same degree sequence.
    """
    if seed is not None:
        np.random.seed(seed)

    permuted = adjacency.copy().tolil()
    rows, cols = adjacency.nonzero()
    edges = list(zip(rows.tolist(), cols.tolist()))
    edge_set = set(edges)
    m = len(edges)

    n_attempts = int(multiplier * m)
    n_swaps = 0

    for _ in range(n_attempts):
        i1 = np.random.randint(m)
        i2 = np.random.randint(m)
        if i1 == i2:
            continue

        a, b = edges[i1]
        c, d = edges[i2]

        # Check: no self-loops, no duplicate edges
        if (a, d) in edge_set or (c, b) in edge_set:
            continue

        # Perform swap
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add((a, d))
        edge_set.add((c, b))

        permuted[a, b] = 0
        permuted[c, d] = 0
        permuted[a, d] = 1
        permuted[c, b] = 1

        edges[i1] = (a, d)
        edges[i2] = (c, b)
        n_swaps += 1

    return permuted.tocsr()


def xswap_null_mean(adjacency_list, source_indices, target_indices,
                    P=200, multiplier=10, verbose=True):
    """
    Compute XSwap null mean for multiple source-target pairs.

    Parameters
    ----------
    adjacency_list : list of scipy.sparse matrices
        One adjacency matrix per edge in the metapath.
    source_indices : array-like
        Source node indices.
    target_indices : array-like
        Target node indices.
    P : int, default=200
        Number of permutations.
    multiplier : int, default=10
        Swap attempts multiplier.
    verbose : bool
        Print progress.

    Returns
    -------
    null_means : ndarray
        Mean path count across P permutations for each pair.
    runtime : float
        Total runtime in seconds.
    """
    import time

    n_pairs = len(source_indices)
    null_counts = np.zeros(n_pairs)
    t0 = time.time()

    for p in range(P):
        # Permute each adjacency matrix
        permuted = [xswap(A, multiplier) for A in adjacency_list]

        # Chain multiply
        product = permuted[0]
        for pm in permuted[1:]:
            product = product @ pm

        product_dense = np.array(product.todense())

        # Count paths for each pair
        for k, (s, t) in enumerate(zip(source_indices, target_indices)):
            null_counts[k] += product_dense[s, t]

        if verbose and (p + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  Permutation {p+1}/{P} ({elapsed:.0f}s)")

    null_means = null_counts / P
    runtime = time.time() - t0

    return null_means, runtime
