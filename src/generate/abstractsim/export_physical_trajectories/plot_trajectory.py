import json
import os

import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python plot_traj.py <log file>")
    #     return


    for i in range(1,20):
        log_file = f'physical_data/traj/trajectories_5_{i}.log'

        with open(log_file, 'r') as f:
            data = json.load(f)

        # Get the robot and ball trajectories
        traj = data['observations']
        robot_traj = [obs[0:3] for obs in traj]
        ball_traj = [obs[3:5] for obs in traj]


        # Convert to numpy arrays
        robot_pos = np.array(robot_traj)
        ball_pos = np.array(ball_traj)

        print(len(robot_pos))


        # Plot the robot and ball trajectories
        plt.plot(robot_pos[:,0], robot_pos[:,1], 'b')
        plt.plot(ball_pos[:,0], ball_pos[:,1], 'r')
        plt.axis([-4500, 4500, -3000, 3000])

        os.makedirs('figures/traj/', exist_ok=True)
        plt.title(f'{i}')
        plt.savefig(f'figures/traj/{i}.png')
        plt.show()

if __name__ == '__main__':
    main()





