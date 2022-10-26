import numpy as np

data_config = dict(
    num_samples=110000,
    num_dim=50,
    margins=0.1,
    width=1.0,
    random_orthonormal_transform=True
)

lms_5_data_config = {**data_config,
                     **{'slabs': np.array([2] + [5] * 49), 'noise_proportions': 0,
                        'slab_probabilities': [[1.0, 1.0]] + [[0.125, 0.5, 0.75, 0.5, 0.125]] * 49}}
ms_57_data_config = {**data_config,
                     **{'slabs': np.array([5] + [7] * 49), 'noise_proportions': 0,
                        'slab_probabilities': [[0.125, 0.5, 0.75, 0.5, 0.125]] +
                                              [[1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25, 1 / 16.0]] * 49}}
lms_7_noisy_data_config = {**data_config,
                           **{'slabs': np.array([2] + [7] * 49),
                              'noise_proportions': np.array([0.1] + [0] * 49),
                              'slab_probabilities': [[1.0, 1.0]] +
                                                    [[1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25, 1 / 16.0]] * 49}}
lms_7_40000_noisy_data_config = {**lms_7_noisy_data_config, 'num_samples': 40000 + 10000}
lms_7_40000_uniform_noisy_data_config = {
    **lms_7_noisy_data_config,
    'slab_probabilities': [[1.0, 1.0]] +
                          [[1.0 / 4, 1.0 / 3, 1.0 / 4, 1.0 / 3, 1.0 / 4, 1.0 / 3, 1.0 / 4]] * 49
}
