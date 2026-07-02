"""
HetNetEX: Exact asymptotic inference for DWPC in heterogeneous knowledge graphs.

Implements Algorithm 1 from:
  Ghosh et al., "HetNetEX: Exact Asymptotic Inference in Heterogeneous
  Biomedical Knowledge Graphs," IEEE BIBM 2026.
"""

import numpy as np
from scipy import stats


class HetNetEX:
    """
    Compute exact asymptotic p-values for degree-weighted path counts (DWPC).

    Parameters
    ----------
    w : float, default=0.4
        Damping exponent. Paths through high-degree nodes are penalized by d^(-w).
        w=0: unweighted path count. w=1: uniform weights. w=0.4: Rephetio standard.

    Examples
    --------
    >>> model = HetNetEX(w=0.4)
    >>> model.add_edge_type('CbG', A_cbg, m_cbg)
    >>> model.add_edge_type('GpPW', A_gpw, m_gpw)
    >>> mu, kappa, sigma, p = model.compute(s, t, ['CbG', 'GpPW'], obs_dwpc)
    """

    def __init__(self, w=0.4):
        self.w = w
        self.edge_types = {}

    def add_edge_type(self, name, adjacency, m=None):
        """
        Register an edge type with its adjacency matrix.

        Parameters
        ----------
        name : str
            Edge type identifier (e.g., 'CbG', 'GiG', 'GpPW').
        adjacency : scipy.sparse matrix
            Adjacency matrix (n_source x n_target).
        m : int, optional
            Number of edges. If None, computed from adjacency.
        """
        if m is None:
            m = adjacency.nnz
        d_source = np.array(adjacency.sum(axis=1)).flatten()
        d_target = np.array(adjacency.sum(axis=0)).flatten()

        # Weighted degree vectors: d^(1-w)
        u = np.power(d_source.astype(float), 1 - self.w)
        v = np.power(d_target.astype(float), 1 - self.w)

        # Degree heterogeneity ratio: <d^2> / (n * <d>^2)
        # Computed for both source and target sides
        eta_source = _heterogeneity_ratio(d_source)
        eta_target = _heterogeneity_ratio(d_target)

        self.edge_types[name] = {
            'adjacency': adjacency,
            'm': m,
            'c': 1.0 / m,
            'd_source': d_source,
            'd_target': d_target,
            'u': u,  # source weighted degrees
            'v': v,  # target weighted degrees
            'eta_source': eta_source,
            'eta_target': eta_target,
            'n_source': len(d_source),
            'n_target': len(d_target),
        }

    def compute_mu(self, source, target, metapath):
        """
        Compute expected null DWPC (Theorem 1) via rank-1 matrix chain.

        Parameters
        ----------
        source : int
            Source node index.
        target : int
            Target node index.
        metapath : list of str
            Sequence of edge type names, e.g., ['CbG', 'GiG', 'GpPW'].

        Returns
        -------
        mu : float
            Expected null DWPC.
        """
        L = len(metapath)
        if L == 0:
            return 0.0

        # Phase 3: chain rank-1 matrices
        et0 = self.edge_types[metapath[0]]
        q = et0['c'] * et0['u'][source] * et0['v']  # vector of length n_target

        for ell in range(1, L):
            et = self.edge_types[metapath[ell]]
            alpha = np.dot(et['u'], q)  # dot product: O(n)
            q = et['c'] * alpha * et['v']  # scalar-vector: O(n)

        # Final: multiply by target node's d^(-w)
        et_last = self.edge_types[metapath[-1]]
        d_t = et_last['d_target'][target]
        if d_t > 0:
            mu = q[target] * (d_t ** (-self.w))
        else:
            mu = 0.0

        return mu

    def compute_kappa(self, mu, metapath):
        """
        Compute overdispersion parameter (Theorem 2).

        Parameters
        ----------
        mu : float
            Expected null DWPC from compute_mu().
        metapath : list of str
            Sequence of edge type names.

        Returns
        -------
        kappa : float
            Overdispersion. kappa=0 means Poisson; kappa>0 means overdispersed.
        """
        L = len(metapath)
        if L <= 1:
            return 0.0

        kappa = 0.0
        for k in range(1, L):
            # Binomial coefficient
            binom = _binom(L - 1, k)
            # Average eta across intermediate positions
            # (simplified: use target-side eta of the k-th edge type)
            et = self.edge_types[metapath[min(k, L - 1)]]
            eta_k = et['eta_target']
            kappa += binom * eta_k * (mu ** (k / (L - 1)))

        return kappa

    def compute(self, source, target, metapath, observed_dwpc):
        """
        Compute p-value for observed DWPC (full Algorithm 1).

        Parameters
        ----------
        source : int
            Source node index.
        target : int
            Target node index.
        metapath : list of str
            Sequence of edge type names.
        observed_dwpc : float
            Observed DWPC value to test.

        Returns
        -------
        mu : float
            Expected null DWPC.
        kappa : float
            Overdispersion parameter.
        sigma : float
            Standard deviation of null distribution.
        p_value : float
            One-sided p-value P(Y >= observed | null).
        """
        mu = self.compute_mu(source, target, metapath)
        kappa = self.compute_kappa(mu, metapath)
        sigma = np.sqrt(max((1 + kappa) * mu, 1e-20))

        if mu < 1e-10:
            p_value = 1.0 if observed_dwpc == 0 else 0.5
        elif mu < 10:
            # Poisson regime (Theorem 3)
            p_value = 1 - stats.poisson.cdf(max(0, int(observed_dwpc) - 1), mu)
        else:
            # CLT regime (Theorem 4)
            z = (observed_dwpc - mu) / sigma
            p_value = 1 - stats.norm.cdf(z)

        return mu, kappa, sigma, p_value

    def compute_all_targets(self, source, metapath):
        """
        Compute mu for all targets simultaneously (vectorized Phase 3).

        Parameters
        ----------
        source : int
            Source node index.
        metapath : list of str
            Sequence of edge type names.

        Returns
        -------
        mu_vector : ndarray
            Expected null DWPC for all targets.
        """
        L = len(metapath)
        et0 = self.edge_types[metapath[0]]
        q = et0['c'] * et0['u'][source] * et0['v']

        for ell in range(1, L):
            et = self.edge_types[metapath[ell]]
            alpha = np.dot(et['u'], q)
            q = et['c'] * alpha * et['v']

        # Apply d^(-w) to all targets
        et_last = self.edge_types[metapath[-1]]
        d_t = et_last['d_target']
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_vector = q * np.where(d_t > 0, d_t ** (-self.w), 0)

        return mu_vector


def _heterogeneity_ratio(degrees):
    """Compute eta = <d^2> / (n * <d>^2). Returns 1.0 for uniform degrees."""
    n = len(degrees)
    d_pos = degrees[degrees > 0]
    if len(d_pos) == 0 or d_pos.mean() == 0:
        return 1.0
    return (d_pos ** 2).mean() / (d_pos.mean() ** 2)


def _binom(n, k):
    """Binomial coefficient C(n, k)."""
    from math import comb
    return comb(n, k)
