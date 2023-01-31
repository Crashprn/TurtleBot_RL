import gym
import numpy as np

from Agent import Agent

env = gym.make("LunarLanderContinuous-v2", render_mode="human")

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
agent.load_models()
agent.eval()

np.random.seed(0)

score_history = []
for i in range(10):
    done = False
    score = 0
    obs = env.reset()[0]
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info, _ = env.step(act)
        score += reward
        obs = new_state
        env.render()

    score_history.append(score)
    print(
        "episode",
        i,
        "score %.2f" % score,
        "100 game average %.2f" % np.mean(score_history[-100:]),
    )
