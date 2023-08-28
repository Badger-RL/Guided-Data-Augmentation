# Memory and disk requirements for each task. The key corresponds to the augmentation ratio, m.
MEMDISK = {
    1: {
        'maze2d-umaze-v1': (1.6, 9),
        'maze2d-medium-v1': (1.6, 9),
        'maze2d-large-v1': (1.6, 9),

        'antmaze-umaze-diverse-v1': (2.2, 9),
        'antmaze-medium-diverse-v1': (2.2, 9),
        'antmaze-large-diverse-v1': (3.3, 9),

        'PushBallToGoal-v0': (1.4, 9),
        'PushBallToGoal-v1': (1.4, 9),
        'PushBallToGoal-v2': (1.4, 9),
        'PushBallToGoalEasy-v0': (1.4, 9),

    },
    2: {
        'maze2d-umaze-v1': (1.9, 9),
        'maze2d-medium-v1': (1.9, 9),
        'maze2d-large-v1': (3.2, 9),
    },
    4: {
        'maze2d-umaze-v1': (2.2, 9),
        'maze2d-medium-v1': (2.2, 9),
        'maze2d-large-v1': (3.5, 9),
    }
}