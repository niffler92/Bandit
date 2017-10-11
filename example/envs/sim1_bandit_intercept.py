import numpy as np

from bandit.bandit import ContextualBandit


class Patient(ContextualBandit):  # Patients keep their context
    def __init__(self, k, d, patient_id, barriers, prev=5):
        assert type(barriers) == np.ndarray
        super().__init__(k, d)
        self.patient_id = patient_id
        self.barriers = barriers.copy()
        self.alphas = self.__set_alpha(self.barriers)
        self.adherence = self.__set_adherence(self.alphas)
        self.memory = {action: np.array([]) for action in range(self.k)}
        self.prev = prev
        self.reset()

    def __set_alpha(self, barriers):
        return np.array([barrier * min(max(np.random.normal(0.65, 0.3), 0), 1)  # alpha = 0.3 in the paper
                         if barrier == 1 else 1 for barrier in barriers])
        # return np.array([barrier * np.random.normal(0.65, 0.3)  # alpha = 0.3 in the paper
        #                  if barrier == 1 else 1 for barrier in barriers])

    def __set_adherence(self, alphas):  # We can change this with logit function later.
        return np.prod(alphas, axis=0)

    def __update_adherence(self, match):
        self.alphas = self.__set_alpha(self.barriers)

        if self.barriers[match]:
            beta = min(max(np.random.normal(0.7, 0.3), 0), 1)
            # beta = np.random.normal(0.7, 0.3)
            self.alphas[match] += beta * (1.0 - self.alphas[match])
        self.action_values = np.multiply(self.alphas, -1)
        self.adherence = np.prod(self.alphas, axis=0)

    def __update_state(self, action, iseffective):
        # Memory를 state의 dimension에 맞게 update 해야
        if self.memory[action].size:
            self.memory[action] = np.vstack((self.memory[action], np.array([int(iseffective)])))
        else:
            self.memory[action] = np.array([int(iseffective)])
        self.states[action] = np.array([1, sum(self.memory[action][-self.prev:])])  # FIXME

    def reset(self):
        self.alphas = self.__set_alpha(self.barriers)
        self.action_values = np.multiply(self.alphas, -1)
        self.adherence = self.__set_adherence(self.alphas)
        self.optimal = np.where((self.alphas < 1) & (self.barriers == 1))[0]
        self.states = np.zeros((self.k, self.d))
        self.memory = {action: np.array([]) for action in range(self.k)}

    def pull(self, action):
        self.__update_adherence(action)
        self.optimal = np.where((self.alphas < 1) & (self.barriers == 1))[0]  # FIXME

        if np.random.rand() < self.adherence:
            self.__update_state(action, True)
            return 1, action in self.optimal
        else:
            self.__update_state(action, False)
            return 0, action in self.optimal

    def __repr__(self):
        return "Patient {} with barriers {}".format(self.patient_id, self.barriers)
