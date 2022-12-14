__all__ = ['LinearSlabDataset']

import numpy as np
from typing import Optional, List, Union, Tuple
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

    def __init__(self, x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray],
                 w: Union[np.ndarray, None], randomized_axes=(), device='cpu'):
        if len(randomized_axes) > 0:
            x = randomize_coordinates(x, w, randomized_axes)
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        if isinstance(y, np.ndarray): y = torch.from_numpy(y)
        x = x.float().to(device=device)
        y = y.long().to(device=device)
        super().__init__(x, y)
        self.w = w
        self.device = device

    def __str__(self):
        return '<{} with {} samples and {} feature dimensions>'.format(
            self.__class__.__name__, self.tensors[0].shape[0], self.tensors[0].shape[1]
        )

    def __repr__(self):
        return self.__str__()

    def randomize_axes(self, axes: Tuple[int]) -> "LinearSlabDataset":
        return LinearSlabDataset(self.tensors[0].cpu().numpy(), self.tensors[1].cpu().numpy(), self.w, axes,
                                 self.device)

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
                 random_orthonormal_transform: bool, device='cpu') -> "LinearSlabDataset":
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

        # compute noises
        if len(noise_proportions.nonzero()[0]) > 0:
            corrupt_mask = rng.uniform(0, 1, size=x.shape) < noise_proportions
            offsets = np.broadcast_to(np.arange(slabs.max() - 1).reshape((-1, 1)), (slabs.max() - 1, num_dim))
            offsets = - width + slab_widths + offsets * (slab_widths + 2 * margins)
            noisy_data = rng.uniform(0, 1, size=x.shape)
            gap_number = rng.integers(0, slabs - 1, size=x.shape)
            noise_offsets = np.take_along_axis(offsets, gap_number, axis=0)
            noisy_data = 2 * margins * noisy_data + noise_offsets
            x[corrupt_mask] = noisy_data[corrupt_mask]

        # just to make it the same as the paper code
        x[:, slabs.reshape((-1,)) == 2] = -x[:, slabs.reshape((-1,)) == 2]

        # transform
        w = np.eye(num_dim)
        if random_orthonormal_transform:
            w = get_orthonormal_matrix(num_dim)
        x = x.dot(w)

        return LinearSlabDataset(x, y, w, device=device)

    def visualize(self, save_as: Optional[str] = None, title='LMS',
                  axis_names=('first component', 'second component', 'third component'),
                  show=True):
        x, y = [t.cpu().numpy() for t in self.tensors]
        w = self.w

        x = x.dot(w.T)
        if self.tensors[0].shape[1] > 2:
            fig, (ax, ax_) = plt.subplots(1, 2, figsize=(16, 5))
            ax.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', s=4, alpha=0.8)
            ax.set_xlabel(axis_names[0], fontsize=15)
            ax.set_ylabel(axis_names[1], fontsize=15)

            ax_.scatter(x[:, 2], x[:, 1], c=y, cmap='coolwarm', s=4, alpha=0.8)
            ax_.set_xlabel(axis_names[2], fontsize=15)
            ax_.set_ylabel(axis_names[1], fontsize=15)
        else:
            fig = plt.figure(figsize=(8, 5))
            plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', s=4, alpha=0.8)
            plt.xlabel(axis_names[0], fontsize=15)
            plt.ylabel(axis_names[1], fontsize=15)
        fig.suptitle(title, fontsize=15)
        if save_as:
            if not os.path.isdir(os.path.dirname(save_as)):
                os.makedirs(os.path.dirname(save_as), exist_ok=True)
            plt.savefig(save_as, bbox_inches='tight')
        if show:
            plt.show()

    def split_train_val(self, train_size):
        indices = torch.randperm(self.tensors[0].shape[0])
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_split = LinearSlabDataset(self.tensors[0][train_indices].contiguous(),
                                        self.tensors[1][train_indices].contiguous(), self.w,
                                        device=self.device)
        val_split = LinearSlabDataset(self.tensors[0][val_indices].contiguous(),
                                      self.tensors[1][val_indices].contiguous(), self.w,
                                      device=self.device)
        return train_split, val_split


def _unit_test():
    data_config = dict(
        num_samples=110000,
        num_dim=50,
        margins=0.1,
        width=1.0,
        random_orthonormal_transform=True
    )
    ms_57_data_config = {**data_config, **{'slabs': np.array([5] + [7] * 49), 'noise_proportions': [0.1] + [0] * 49,
                                           'slab_probabilities': [[0.125, 0.5, 0.75, 0.5, 0.125]] +
                                                                 [[1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25,
                                                                   1 / 16.0]] * 49}}

    ms_57_data = LinearSlabDataset.generate(**ms_57_data_config)


if __name__ == '__main__':
    _unit_test()
