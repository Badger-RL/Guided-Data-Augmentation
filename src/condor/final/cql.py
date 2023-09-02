PARAMS = {
    'antmaze-umaze-diverse-v1': {
        'no_aug': (64, 64, 3e-5, 3e-5, 5),
        'random': (256, 64, 3e-5, 3e-5, 2.5),
        'guided': (256, 64, 3e-4, 3e-4, 5),
    },
    'antmaze-medium-diverse-v1': {
        'no_aug': (64, 64, 3e-5, 3e-5, 0.5),
        'random': (64, 64, 3e-5, 3e-5, 0.5), # good
        'guided': (64, 64, 3e-5, 3e-5, 0.5),
    },
    'antmaze-large-diverse-v1': {
        'no_aug': (64, 64, 3e-4, 3e-4, 2.5),
        'random': (64, 64, 3e-5, 3e-5, 10),
        'guided': (64, 64, 3e-4, 3e-4, 10),
    },
    'PushBallToGoal-v0': {
        'no_aug': {
            'n_layers': 2,
            'hidden_dims': 256
        },
        'random': (64, 64, 3e-5, 3e-5, 10),
        'guided': (64, 64, 3e-4, 3e-4, 10),
    },
}
