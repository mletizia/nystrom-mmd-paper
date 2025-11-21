import numpy as np

def MMD2b_from_K(K: np.ndarray, n_x: int) -> float:
    """
    Compute the biased MMD^2 statistic between two groups, given a
    full kernel matrix and the size n_x of group X.
    Group X is assumed to be indices {0, 1, ..., n_x-1}.
    Group Y is the complement.
    """
    N = K.shape[0]

    # Construct index set for X
    A = np.arange(n_x, dtype=int)

    # Construct index set for Y (complement of A)
    mask = np.ones(N, dtype=bool)
    mask[:n_x] = False
    B = np.nonzero(mask)[0]

    n_y = N - n_x

    K_xx = K[np.ix_(A, A)]
    K_yy = K[np.ix_(B, B)]
    K_xy = K[np.ix_(A, B)]

    sum_xx = K_xx.sum()
    sum_yy = K_yy.sum()
    sum_xy = K_xy.sum()

    return (
        (sum_xx / (n_x ** 2))
        + (sum_yy / (n_y ** 2))
        - (2.0 * sum_xy / (n_x * n_y))
    )
