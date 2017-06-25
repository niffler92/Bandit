import numpy as np


class Agent(object):
    def __init__(self, k, policy, prior=0, gamma=None):
        self.policy = policy
        self.k = k
        self.prior = prior
        self.gamma = gamma
        self._value_estimates = prior * np.ones(self.k)  # Estimated Mean reward
        self.action_attempts = np.zeros(self.k)
        self.t = 0
        self.last_action = None

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._value_estimates[:] = self.prior * np.ones(self.k)
        self.action_attempts[:] = np.zeros(self.k)
        self.last_action = None
        self.t = 0

    def choose(self, random_period):
        if self.t < random_period:
            action = np.random.randint(self.k)
        else:
            action = self.policy.choose(self)
        self.last_action = action

        return action

    def observe(self, reward):  # Updating value_estimates ! (calculating mean rewards)
        self.action_attempts[self.last_action] += 1

        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action]
        else:
            g = self.gamma
        q = self._value_estimates[self.last_action]

        self._value_estimates[self.last_action] += g * (reward - q)
        self.t += 1

    @property
    def value_estimates(self):
        return self._value_estimates


class ContextualAgent(Agent):
    """
    ( for linUCB disjoint model)
    """

    def __init__(self, k, d, policy, prior=0, gamma=None):
        super().__init__(k, policy, prior, gamma)
        self.d = d
        self.memory = {action: {'A': np.identity(self.d), 'b': np.zeros((self.d, 1))} for action in range(self.k)}
        self.states = None
        self.reset()

    def reset(self):
        self._value_estimates[:] = self.prior * np.ones(self.k)
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0
        self.memory = {action: {'A': np.identity(self.d), 'b': np.zeros((self.d, 1))} for action in range(self.k)}
        self.states = None

    # FIXME
    # choose_bandit and get state from memory
    def get_state(self, bandit):
        self.states = bandit.states
        for action, memory in self.memory.items():
            A = memory['A']
            b = memory['b']
            A_inv = np.linalg.inv(A)
            theta_hat = np.dot(A_inv, b)
            x_t = self.states[action]
            self._value_estimates[action] = np.dot(x_t.T, theta_hat)

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1
        self.memory[self.last_action]['A'] += np.outer(self.states[self.last_action],
                                                       self.states[self.last_action])
        if self.memory[self.last_action]['b'].size:
            self.memory[self.last_action]['b'] += reward * self.states[self.last_action]
        else:
            self.memory[self.last_action]['b'] = reward * self.states[self.last_action]

        self.t += 1
