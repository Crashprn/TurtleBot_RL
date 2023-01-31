import numpy as np


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None) -> None:
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.mode_arg = "train"
        self.reset()

    def __call__(self):
        if self.mode_arg == "train":
            x = (
                self.x_prev
                + self.theta * (self.mu - self.x_prev) * self.dt
                + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
            )

            self.x_prev = x
            return x
        else:
            return 0

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def eval(self):
        self.mode_arg = "eval"

    def train(self):
        self.mode_arg = "train"
