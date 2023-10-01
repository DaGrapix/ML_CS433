import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    e = y - tx@w
    g = np.zeros_like(e)
    g[e>=0] = 1
    g[e<0]  = -1

    subgradient = -(1/len(y))*tx.T@g
    return subgradient
