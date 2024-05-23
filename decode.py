from pyldpc import utils
import numpy as np
import warnings

from numba import njit, int64, types, float64, int32, cuda


def decode(H, y, snr, maxiter=1000):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm.

    Decoding is performed in parallel if multiple codewords are passed in y.

    Parameters
    ----------
    H: array (n_equations, n_code). Decoding matrix H.
    y: array (n_code, n_messages) or (n_code,). Received message(s) in the
        codeword space.
    maxiter: int. Maximum number of iterations of the BP algorithm.

    Returns
    -------
    x: array (n_code,) or (n_code, n_messages) the solutions in the
        codeword space.

    """
    m, n = H.shape

    bits_hist, bits_values, nodes_hist, nodes_values = utils._bitsandnodes(H)

    solver = _logbp_numba

    var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]
    # step 0: initialization

    Lc = 2 * y / var
    _, n_messages = y.shape

    Lq = np.zeros(shape=(m, n, n_messages))

    Lr = np.zeros(shape=(m, n, n_messages))
    for n_iter in range(maxiter):
        L_posteriori = np.zeros((n, n_messages))
        solver(bits_hist, bits_values, nodes_hist,
               nodes_values, Lc, Lq, Lr, n_iter, L_posteriori)
        x = np.array(L_posteriori <= 0).astype(int)
        product = utils.incode(H, x)
        if product:
            break
    if n_iter == maxiter - 1:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x.squeeze()


def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter, L_posteriori):
    """Perform inner ext LogBP solver."""
    m, n, n_messages = Lr.shape
    # step 1 : Horizontal
    bits_counter = 0
    nodes_counter = 0
    for i in range(m):
        # ni = bits[i]
        ff = bits_hist[i]
        ni = bits_values[bits_counter: bits_counter + ff]
        bits_counter += ff
        for j in ni:
            nij = ni[:]

            X = np.ones(n_messages)
            if n_iter == 0:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])
            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])

    # step 2 : Vertical
    for j in range(n):
        # mj = nodes[j]
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]

            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
    nodes_counter = 0
    for j in range(n):
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)
