import numpy as np


class MultiArmedBandit(object):
    """
    Multi-armed single Bandit
    Args
        k: number of arms
    """

    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = None

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = None

    def pull(self, action):
        return 0, True  # returns reward and True if action is optimal.


class ContextualBandit(MultiArmedBandit):
    """
    Usually it is normal to have the agent keep the state of each bandit.
    But in some cases it is easier to have bandit keep their own state.
    Args
        k: number of arms
        d: dimension of state vector given action
    """

    def __init__(self, k, d):  # d: dimension of state
        super().__init__(k)
        self.d = d
        self.states = np.zeros((self.k, self.d))

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0
        self.states = np.zeros((self.k, self.d))


class MultiBandits(object):
    def __init__(self):
        self.bandits = []
        self.bandit = None
        self.cursor = 0
        self.k = 0
        self.reset()

    def reset(self):
        self.bandits = []
        self.bandit = None
        self.cursor = 0
        self.k = 0

    def add_bandit(self, bandit):
        self.bandits.append(bandit)
        self.k = bandit.k

    def get_bandit(self):
        self.bandit = self.bandits[self.cursor]
        self.k = self.bandit.k
        self.cursor += 1
        if self.cursor == len(self.bandits):
            self.cursor = 0

    def pull(self, action):
        return self.bandit.pull(action)
