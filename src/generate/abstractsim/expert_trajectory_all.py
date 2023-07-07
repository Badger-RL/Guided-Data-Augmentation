import os

if __name__ == "__main__":

    # os.chdir('generate')
    for i in range(10):
        os.system(
            f'python ./expert_trajectory.py --seed {i+42}'
            f' --save-dir ../../datasets/PushBallToGoal-v0/trajectories'
            f' --save-name {i}.hdf5')