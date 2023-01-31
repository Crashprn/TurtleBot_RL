import numpy as np
import torch as T

from TurtleBot_RL import Agent
from TurtleBot_RL import Simulation

def main():
    print(__file__ + " start!!")
    grid_size = 2  # [m]
    robot_radius = 1.0  # [m]
    obstacle_count = 5

    sim = Simulation(robot_radius, grid_size, obstacle_count)

    agent = Agent(
        alpha=0.000025,
        beta=0.00025,
        input_dims=[12],
        tau=0.001,
        batch_size=64,
        fc1_dims=300,
        fc2_dims=200,
        fc3_dims=100,
        n_actions=2,
        action_range=1,
        run_name="2"
    )
    agent.load_models()
    print("Using GPU: {}".format(T.cuda.is_available()))

    score_history = []
    for i in range(1000):
        done = False
        score = 0
        obs = sim.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done = sim.step(act)
            #agent.remember(obs, act, reward, new_state, int(done))
            #agent.learn()
            score += reward
            obs = new_state
            if score < -8000:
                break

        score_history.append(score)
        print(
            "episode",
            i,
            "score %.2f" % score,
            "100 game average %.2f" % np.mean(score_history[-100:]),
        )

        # if i % 25 == 0:
        #     agent.save_models()
        #     sim.plot_res(score_history, "Learning Rate Graphs", 2000, int(sys.argv[1]))
        if i % 1 == 0:
            sim.showPath()





if __name__ == '__main__':
    main()
