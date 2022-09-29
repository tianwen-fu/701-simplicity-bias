__all__ = ['generate_lms_array', 'visualize_lms_array', 'save_arrays', 'load_arrays']

import numpy as np
from typing import Tuple
from collections.abc import Iterable

import torch
from scipy.linalg import qr
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

rng = np.random.default_rng()


def get_orthonormal_matrix(n):
    # copied from https://github.com/harshays/simplicitybiaspitfalls/blob/master/scripts/utils.py
    H = np.random.randn(n, n)
    s = np.linalg.svd(H)[1]
    s = s[s > 1e-7]
    if len(s) != n: return get_orthonormal_matrix(n)
    Q, R = qr(H)
    return Q


def generate_lms_array(num_samples, num_dim, width, slabs: np.ndarray, margins: float | np.ndarray,
                       noise_proportions: float | np.ndarray,
                       random_orthonormal_transform: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ndarrays for LMS-n or MS-(n, m) data
    :param num_samples: number of samples in the array
    :param num_dim: (D) number of dimensions of the generated data
    :param width: the range of data [-W, W]
    :param slabs: (D,) array, number of slabs for each dimension
    :param margins: (D,) array or single number, margin between two consecutive slabs
    :param noise_proportions: (D,) proportion of data to be corrupted, only work for linear
    :param random_orthonormal_transform: whether multiply a random orthonormal matrix
    :return: X (N, D), y (N,), w (D, D)
    """
    if not isinstance(margins, Iterable):
        margins = np.full((num_dim,), margins, dtype=np.float32)
    if not isinstance(noise_proportions, Iterable):
        noise_proportions = np.full_like(margins, noise_proportions, shape=(num_dim,))
    assert slabs.shape == (num_dim,)
    assert (slabs > 1).all()
    assert ((slabs == 2) | (noise_proportions == 0)).all()
    slabs = slabs.reshape((1, num_dim))
    margins = margins.reshape((1, num_dim))
    noise_proportions = noise_proportions.reshape((1, num_dim))
    slab_widths = (2 * width - 2 * (slabs - 1) * margins) / slabs
    assert (slab_widths > 0).all()

    x = rng.uniform(0.0, 1.0, size=(num_samples, num_dim))
    y = rng.choice([0, 1], size=(num_samples, 1))  # labels
    n_positive_slabs = slabs // 2
    n_negative_slabs = slabs - n_positive_slabs
    n_slabs = n_positive_slabs * y + n_negative_slabs * (1 - y)  # (N, D)
    slab_no = rng.integers(0, n_slabs)
    slab_no = slab_no * 2 + y  # 0th negative slab: 0th slab; 0th positive slab: 1st slab
    slab_lower_bound = - width + slab_no * (slab_widths + margins)
    x = x * slab_widths + slab_lower_bound
    y = y.reshape((num_samples,))

    corrupt_mask = rng.uniform(0, 1, size=x.shape) < noise_proportions
    noisy_data = rng.uniform(-margins, margins, size=x.shape)
    x[corrupt_mask] = noisy_data[corrupt_mask]

    # transform
    w = np.eye(num_dim)
    if random_orthonormal_transform:
        w = get_orthonormal_matrix(num_dim)
    x = x.dot(w)

    return x, y, w


def visualize_lms_array(x, y, w, save=None, title='LMS'):
    fig, (ax, ax_) = plt.subplots(1, 2, figsize=(16, 5))
    x = x.dot(w.T)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', s=4, alpha=0.8)
    ax.set_xlabel('first component', fontsize=15)
    ax.set_ylabel('second component', fontsize=15)

    ax_.scatter(x[:, 2], x[:, 1], c=y, cmap='coolwarm', s=4, alpha=0.8)
    ax_.set_xlabel('second component', fontsize=15)
    ax_.set_ylabel('third component', fontsize=15)
    fig.suptitle(title, fontsize=15)
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()


def save_arrays(filename: str, x: np.ndarray, y: np.ndarray, w: np.ndarray | None):
    np.savez(filename, x=x, y=y, w=w)


def load_arrays(filename) -> Tuple[TensorDataset, np.ndarray | None]:
    data = np.load(filename)
    x = torch.from_numpy(data['x'])
    y = torch.from_numpy(data['y'])
    w = torch.from_numpy(data['w'])
    return TensorDataset(x, y), w
