import numpy as np


class Policy(object):
    """
    A policy decides an action with information of the agent
    """

    def choose(self, agent):
        pass


class UCBPolicy(Policy):
    """
    UCB Policy chooses action which maximizes Upper Confidence Bound
    """

    def __init__(self, c=2):
        self.c = c

    def choose(self, agent):
        exploration = 2 * np.log(agent.t + 1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1 / self.c)

        ucb = agent.value_estimates + exploration
        action = np.argmax(ucb)
        return action


class LinUCBPolicy(UCBPolicy):
    """
    LinUCB assumes expected reward is a linear combination of state(context)
    given an action.
    """

    def __init__(self, alpha, d):
        self.alpha = alpha
        self.d = d

    def choose(self, agent):
        ucb = np.array([])

        for action, memory in agent.memory.items():
            A_inv = np.linalg.inv(memory['A'])
            x_t = agent.states[action]
            exploration = self.alpha * np.sqrt(np.dot(np.dot(x_t.T, A_inv), x_t))
            ucb = np.append(ucb, agent.value_estimates[action] + exploration)

        ucb[np.random.choice(len(ucb))] += 0.000001  # For tie values
        action = np.argmax(ucb)

        return action


class RandomPolicy(Policy):
    def choose(self, agent):
        return np.random.randint(agent.k)
