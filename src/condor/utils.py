# Memory and disk requirements for each task. The key corresponds to the augmentation ratio, m.
MEMDISK = {
    'small': {
        'maze2d-umaze-v1': (1.5, 9),
        'maze2d-medium-v1': (1.5, 9),
        'maze2d-large-v1': (1.5, 9),
        'PushBallToGoal-v0': (1.5, 9),

        'antmaze-umaze-diverse-v1': (1.5, 9),
        'antmaze-medium-diverse-v1': (1.5, 9),
        'antmaze-large-diverse-v1': (1.5, 9),

    },
    'large': {
        'maze2d-umaze-v1': (2.2, 9),
        'maze2d-medium-v1': (2.2, 9),
        'maze2d-large-v1': (2.2, 9),
        'PushBallToGoal-v0': (2.2, 9),

        'antmaze-umaze-diverse-v1': (2.2, 9),
        'antmaze-medium-diverse-v1': (2.2, 9),
        'antmaze-large-diverse-v1': (2.2, 9),
    }
}