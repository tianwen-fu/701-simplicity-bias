from copy import deepcopy

__all__ = ['lms_7_fcn300_2', 'lms_5_fcn300_2']

SLAB_PROB_7 = (1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25, 1 / 16.0)
SLAB_PROB_5 = (0.125, 0.5, 0.75, 0.5, 0.125)

lms_7_fcn300_2 = dict(
    data=dict(
        num_dim=50,
        margins=0.1,
        width=1.0,
        random_orthonormal_transform=True,
        slabs=(dict(count=1, val=2), dict(count=49, val=7)),
        noise_proportions=(dict(count=1, val=0.1), dict(count=49, val=0)),
        slab_probabilities=(dict(count=1, val=(1.0, 1.0)), dict(count=49, val=SLAB_PROB_7)),

        # configs for the runner:
        train_samples=50000,
        val_samples=10000,
        simple_axes=(0,),
        batch_size=256,
    ),
    model=dict(
        cls='fcn',
        num_layers=2,
        input_dim=50,
        output_dim=2,
        latent_dim=300,
        use_bn=False,
        dropout_probability=0.0,
        linear_init=None,
        loss='CrossEntropy',
    ),
    optimizer=dict(
        cls='SGD',
        lr=0.3,
        weight_decay=5.0e-4,
        momentum=0.0
    ),
    trainer=dict(
        premature_evaluate_interval=1000,
        evaluate_interval=5000,
        save_interval=0,
        accuracy_threshold=0.995,
        max_steps=250000,
    ),
)

lms_5_fcn300_2 = deepcopy(lms_7_fcn300_2)
lms_5_fcn300_2['data']['slabs'] = (dict(count=1, val=2), dict(count=49, val=5))
lms_5_fcn300_2['data']['slab_probabilities'] = (dict(count=1, val=(1.0, 1.0)), dict(count=49, val=SLAB_PROB_5))
