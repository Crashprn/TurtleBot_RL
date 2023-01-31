import gym
import numpy as np
import torch as T
from Agent import Agent

env = gym.make("LunarLander-v2", continuous=True)

agent = Agent(
    alpha=0.000025,
    beta=0.00025,
    input_dims=[8],
    tau=0.001,
    batch_size=64,
    fc1_dims=400,
    fc2_dims=300,
    n_actions=2,
)


print(T.cuda.is_available())
np.random.seed(0)

score_history = []
for i in range(1000):
    done = False
    score = 0
    obs = env.reset()[0]
    while not done:
        # Get the best action given the state from the agent
        act   = agent.choose_action(obs)
        # Use the action to get the new state of the agent
        new_state, reward, done, info, _ = env.step(act)
        # Have agent remember its prev_state, action, reward, new state, and whether the episode is done
        agent.remember(obs, act, reward, new_state, int(done))
        # Have agent learn the reward for the particular action and state. Plus learn to output action
        agent.learn()

        score += reward
        obs = new_state

    score_history.append(score)
    print(
        "episode",
        i,
        "score %.2f" % score,
        "100 game average %.2f" % np.mean(score_history[-100:]),
    )

    if i % 25 == 0:
        agent.save_models()
