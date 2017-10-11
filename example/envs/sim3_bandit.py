import numpy as np

from bandit.bandit import ContextualBandit


class Patient(ContextualBandit):  # Patients keep their context
    def __init__(self, k, d, patient_id, barriers):
        assert type(barriers) == np.ndarray
        super().__init__(k, d)
        self.patient_id = patient_id
        self.barriers = barriers.copy()
        self.alphas = self.__set_alpha(self.barriers)
        self.adherence = self.__set_adherence(self.alphas)
        self.memory = {action: np.array([]) for action in range(self.k)}
        self.fatigue_fct = 1
        self.msg_sent = np.zeros(shape=(k - 1, 1))
        self.msg_sent_prev2 = False
        self.isLingering = False
        self.reset()

    def __set_alpha(self, barriers):
        return np.array([barrier * np.random.normal(0.65, 0.3)  # alpha = 0.3 in the paper
                         if barrier == 1 else 1 for barrier in barriers])

    def __set_adherence(self, alphas):  # We can change this with logit function later.
        return np.prod(alphas, axis=0)

    def __update_adherence(self, match):
        self.fatigue_fct = (max(self.fatigue_fct - 0.05, 0) if self.msg_sent_prev2
                            else min(self.fatigue_fct + 0.05, 1))
        self.alphas = self.__set_alpha(self.barriers)

        pos = np.where(np.apply_along_axis(np.prod, axis=1, arr=self.msg_sent[:, -3:-1]) == 1)[0]
        self.isLingering = True if pos.size and self.barriers[pos[0]] == 1 and match == 3 else False

        beta = np.random.normal(loc=0.7, scale=0.3)
        if match == 3:
            if self.isLingering:
                self.alphas[pos] += 0.5 * beta * (1.0 - self.alphas[pos])
        elif self.barriers[match]:
            self.alphas[match] += self.fatigue_fct * beta * (1.0 - self.alphas[match])

        self.action_values = np.multiply(self.alphas, -1)
        self.adherence = np.prod(self.alphas, axis=0)

    def __update_state(self, action, iseffective):
        # Memory를 state의 dimension에 맞게 update 해야
        if self.memory[action].size:
            self.memory[action] = np.vstack((self.memory[action], np.array([int(iseffective)])))
        else:
            self.memory[action] = np.array([int(iseffective)])

        # FIXME
        self.msg_sent_prev2 = np.prod(
                np.apply_along_axis(
                    np.any, axis=0, arr=self.msg_sent[:, -2:]))
        self.states[action] = np.array([1, sum(self.memory[action][-5:]), int(self.msg_sent_prev2)])
        self.states[:, -1] = int(self.msg_sent_prev2)

    def reset(self):
        self.alphas = self.__set_alpha(self.barriers)
        self.action_values = np.multiply(self.alphas, -1)
        self.adherence = self.__set_adherence(self.alphas)
        self.optimal = np.where((self.alphas <= 1) & (self.barriers == 1))[0]
        self.states = np.zeros((self.k, self.d))
        self.memory = {action: np.array([]) for action in range(self.k)}
        self.msg_sent = np.zeros(shape=((self.k - 1), 1))
        self.fatigue_fct = 1

    def pull(self, action):

        if action == 3:
            self.msg_sent = np.hstack((self.msg_sent, np.zeros(shape=(self.k - 1, 1))))
        else:
            action_col = np.zeros(shape=(self.k - 1, 1))
            action_col[action] = 1
            self.msg_sent = np.hstack((self.msg_sent, action_col))

        self.__update_adherence(action)
        self.optimal = np.where(self.barriers == 1)[0]

        if np.random.rand() < self.adherence:
            self.__update_state(action, True)
            return 1, action in self.optimal
        else:
            self.__update_state(action, False)
            return 0, action in self.optimal

    def __repr__(self):
        return "Patient {} with barriers {}".format(self.patient_id, self.barriers)
