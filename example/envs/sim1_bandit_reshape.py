import numpy as np

from bandit.bandit import ContextualBandit


class Patient(ContextualBandit):  # Patients keep their context
    def __init__(self, k, d, patient_id, barriers, prev=5):
        assert type(barriers) == np.ndarray
        super().__init__(k, d)
        self.patient_id = patient_id
        self.barriers = barriers.copy()
        self.alphas = self.__set_alpha(self.barriers)
        self.cutoff = 0.5
        self.sig1 = 1
        self.sig2 = 1
        self.e1 = np.random.normal(0, self.sig1)
        self.engagement = self.__set_engagement(self.e1, self.sig2)
        self.adherence = self.__set_adherence(self.alphas)
        self.memory = {action: np.array([]) for action in range(self.k)}
        self.prev = prev
        self.reset()

    @staticmethod
    def __set_alpha(barriers):
        return np.array([barrier * np.random.normal(0.65, 0.03)  # alpha = 0.3 in the paper
                         if barrier == 1 else 1 for barrier in barriers])

    @staticmethod
    def __set_engagement(e1, sig2):
        e2 = np.random.normal(loc=0, scale=sig2)
        return np.exp(e1 + e2) / (1 + np.exp(e1 + e2))

    def __set_adherence(self, alphas):
        return np.prod(alphas, axis=0) * self.engagement

    def __update_adherence(self, match):
        self.alphas = self.__set_alpha(self.barriers)

        if self.barriers[match]:
            beta = np.random.normal(loc=0.7, scale=0.03)
            self.alphas[match] += beta * (1.0 - self.alphas[match])
        self.adherence = np.prod(self.alphas, axis=0)

    def __update_state(self, action, iseffective):
        if self.memory[action].size:
            self.memory[action] = np.vstack((self.memory[action], np.array([int(iseffective)])))
        else:
            self.memory[action] = np.array([int(iseffective)])
        self.states[action] = np.array([1, sum(self.memory[action][-self.prev:])])  # FIXME

    def reset(self):
        self.alphas = self.__set_alpha(self.barriers)
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
