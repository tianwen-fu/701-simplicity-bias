__all__ = ['LinearSlabDataset']

import numpy as np
from typing import Optional, List, Union
from collections.abc import Iterable
import torch
from scipy.linalg import qr
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import os

rng = np.random.default_rng()


def get_orthonormal_matrix(n):
    # copied from https://github.com/harshays/simplicitybiaspitfalls/blob/master/scripts/utils.py
    H = np.random.randn(n, n)
    s = np.linalg.svd(H)[1]
    s = s[s > 1e-7]
    if len(s) != n: return get_orthonormal_matrix(n)
    Q, R = qr(H)
    return Q


def randomize_coordinates(x, w, axes):
    if w is None:
        w = np.eye(x.shape[1])
    x = np.dot(x, w.T)
    for ax in axes:
        p = rng.permutation(x.shape[0])
        x[:, ax] = x[p, ax]
    x = np.dot(x, w)

    return x


class LinearSlabDataset(torch.utils.data.TensorDataset):
    w: Union[np.ndarray, None]

    def __init__(self, x: np.ndarray, y: np.ndarray, w: Union[np.ndarray, None], randomized_axes=()):
        if len(randomized_axes) > 0:
            x = randomize_coordinates(x, w, randomized_axes)
        super().__init__(torch.from_numpy(x), torch.from_numpy(y))
        self.w = w

    def save_as(self, data_path: str, train_split: int = 0):
        """
        Save the dataset
        :param data_path: path to the saved file, do not include extension '.npz'
        :param train_split: number of training samples if to split. "0" for no splitting.
        """
        x, y = self.tensors
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        if train_split > 0:
            indices = np.arange(x.shape[0])
            rng.shuffle(indices)
            x = x[indices]
            y = y[indices]
            np.savez(data_path + '_train.npz', x=x[:train_split], y=y[:train_split], w=self.w)
            np.savez(data_path + '_val.npz', x=x[train_split:], y=y[train_split:], w=self.w)
        else:
            np.savez(data_path, x=x, y=y, w=self.w)

    @staticmethod
    def from_file(data_path: str, **kwargs):
        data = np.load(data_path)
        x, y, w = data['x'], data['y'], data['w']
        return LinearSlabDataset(x, y, w, **kwargs)

    @staticmethod
    def generate(num_samples, num_dim, width, slabs: np.ndarray, margins: Union[float, np.ndarray],
                 noise_proportions: Union[float, np.ndarray], slab_probabilities: List[List[float]],
                 random_orthonormal_transform: bool) -> "LinearSlabDataset":
        """
        Generate ndarrays for LMS-n or MS-(n, m) data
        Currently only supporting noise on linear slabs
        :param num_samples: number of samples in the array
        :param num_dim: (D) number of dimensions of the generated data
        :param width: the range of data [-W, W]
        :param slabs: (D,) array, number of slabs for each dimension (always P-N-P-N-...)
        :param margins: (D,) array or single number, margin between two consecutive slabs
        :param noise_proportions: (D,) proportion of data to be corrupted, only work for linear
        :param slab_probabilities: probabilities for sampling each slab, pad the non-existent arrays with 0
        :param random_orthonormal_transform: whether multiply a random orthonormal matrix
        :return: X (N, D), y (N,), w (D, D)
        """
        margins = np.broadcast_to(margins, (num_dim,)).astype(np.float32)
        noise_proportions = np.broadcast_to(noise_proportions, (num_dim,)).astype(np.float32)
        assert slabs.shape == (num_dim,)
        assert (slabs > 1).all()
        assert ((slabs == 2) | (noise_proportions == 0)).all()
        slabs = slabs.reshape((1, num_dim))
        margins = margins.reshape((1, num_dim))
        noise_proportions = noise_proportions.reshape((1, num_dim))
        slab_widths = (2 * width - 2 * (slabs - 1) * margins) / slabs
        assert (slab_widths > 0).all()
        assert all(sum(probs[::2]) == 1.0 and sum(probs[1::2]) == 1.0 for probs in slab_probabilities)

        x = rng.uniform(0.0, 1.0, size=(num_samples, num_dim))
        y = rng.choice([0, 1], size=(num_samples, 1))  # labels
        n_negative_slabs = slabs // 2
        n_positive_slabs = slabs - n_negative_slabs
        positive_slab_no = np.stack([
            rng.choice(np.arange(n), p=p[::2], size=(num_samples,))
            for n, p in zip(n_positive_slabs.flatten(), slab_probabilities)
        ], axis=1)
        negative_slab_no = np.stack([
            rng.choice(np.arange(n), p=p[1::2], size=(num_samples,))
            for n, p in zip(n_negative_slabs.flatten(), slab_probabilities)
        ], axis=1)
        assert positive_slab_no.shape == (num_samples, num_dim)
        # 0th positive slab: 0th slab; 0th negative slab: 1st slab
        slab_no = positive_slab_no * 2 * y + (negative_slab_no * 2 + 1) * (1 - y)
        slab_lower_bound = - width + slab_no * (slab_widths + margins * 2)
        x = x * slab_widths + slab_lower_bound
        y = y.reshape((num_samples,))

        corrupt_mask = rng.uniform(0, 1, size=x.shape) < noise_proportions
        noisy_data = rng.uniform(-margins, margins, size=x.shape)
        x[corrupt_mask] = noisy_data[corrupt_mask]
        # just to make it the same as the paper code
        x[:, slabs.reshape((-1,)) == 2] = -x[:, slabs.reshape((-1,)) == 2]

        # transform
        w = np.eye(num_dim)
        if random_orthonormal_transform:
            w = get_orthonormal_matrix(num_dim)
        x = x.dot(w)

        return LinearSlabDataset(x, y, w)

    def visualize(self, save_as: Optional[str] = None, title='LMS',
                  axis_names=('first component', 'second component', 'third component')):
        x, y = [t.cpu().numpy() for t in self.tensors]
        w = self.w

        fig, (ax, ax_) = plt.subplots(1, 2, figsize=(16, 5))
        x = x.dot(w.T)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', s=4, alpha=0.8)
        ax.set_xlabel(axis_names[0], fontsize=15)
        ax.set_ylabel(axis_names[1], fontsize=15)

        ax_.scatter(x[:, 2], x[:, 1], c=y, cmap='coolwarm', s=4, alpha=0.8)
        ax_.set_xlabel(axis_names[2], fontsize=15)
        ax_.set_ylabel(axis_names[1], fontsize=15)
        fig.suptitle(title, fontsize=15)
        if save_as:
            if not os.path.isdir(os.path.dirname(save_as)):
                os.makedirs(os.path.dirname(save_as), exist_ok=True)
            plt.savefig(save_as, bbox_inches='tight')
        plt.show()
