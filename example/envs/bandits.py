from datetime import datetime

import numpy as np
import pandas as pd

from bandit.bandit import ContextualBandit


class Patient(ContextualBandit):  # Patients keep their context
    def __init__(self, k, d, patient_id, barriers, prev=5, use_intercept=True):
        assert type(barriers) == np.ndarray
        super().__init__(k, d)
        self.patient_id = patient_id
        self.barriers = barriers.copy()
        self.memory = {action: np.array([]) for action in range(self.k)}
        self.prev = prev
        self.alphas = self._set_alphas()
        self.use_intercept = use_intercept
        self.reset()

    @property
    def adherence(self):
        return np.prod(self.alphas, axis=0)

    def _set_alphas(self):
        return np.array([barrier * np.random.normal(0.65, 0.1)
                         if barrier == 1 else 1 for barrier in self.barriers])

    def _update_adherence(self, match):
        self.alphas = self._set_alphas()

        if self.barriers[match]:
            beta = np.random.normal(loc=0.7, scale=0.3)
            self.alphas[match] += beta * (1.0 - self.alphas[match])

    def _update_state(self, action, iseffective):
        if self.memory[action].size:
            self.memory[action] = np.vstack((self.memory[action],
                                             np.array([int(iseffective)])))
        else:
            self.memory[action] = np.array([int(iseffective)])

        if self.use_intercept:
            self.states[action] = np.array([1, sum(self.memory[action][-self.prev:])])  # FIXME
        else:
            self.states[action] = np.array([sum(self.memory[action][-self.prev:])])

    def reset(self):
        self.alphas = self._set_alphas()
        self.optimal = np.where(np.logical_and(
            (self.alphas < 1), (self.barriers == 1)))[0]
        self.states = np.zeros((self.k, self.d))
        self.memory = {action: np.array([]) for action in range(self.k)}

    def pull(self, action):
        # self.action = action
        self._update_adherence(action)
        self.optimal = np.where((self.alphas < 1) & (self.barriers == 1))[0]  # FIXME

        if np.random.rand() < self.adherence:
            self._update_state(action, True)
            return 1, action in self.optimal
        else:
            self._update_state(action, False)
            return 0, action in self.optimal

    def __repr__(self):
        return "Patient {} with barriers {}".format(
            self.patient_id, self.barriers)


class PatientFatigue(Patient):
    def __init__(self, k, d, patient_id, barriers, use_intercept=True):
        assert type(barriers) == np.ndarray
        super().__init__(k, d)
        self.fatigue_fct = 1
        self.msg_sent = np.zeros(shape=(k - 1, 1))
        self.msg_sent_prev2 = False
        self.isLingering = False

    def _update_adherence(self, match):
        self.fatigue_fct = (max(self.fatigue_fct - 0.05, 0) if self.msg_sent_prev2
                            else min(self.fatigue_fct + 0.05, 1))
        self.alphas = self._set_alphas(self.barriers)

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

    def _update_state(self, action, iseffective):
        if self.memory[action].size:
            self.memory[action] = np.vstack((self.memory[action], np.array([int(iseffective)])))
        else:
            self.memory[action] = np.array([int(iseffective)])

        # FIXME
        self.msg_sent_prev2 = np.prod(
                np.apply_along_axis(
                    np.any, axis=0, arr=self.msg_sent[:, -2:]))

        if self.use_intercept:
            self.states[action] = np.array([1, sum(self.memory[action][-self.prev:]), int(self.msg_sent_prev2)])
        else:
            self.states[action] = np.array([sum(self.memory[action][-self.prev:]), int(self.msg_sent_prev2)])

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


class PatientEngagement(Patient):
    """
    Args
        k: How many actions
        d: dimension of states
    """
    def __init__(self, k, d, patient_id, barriers, prev=5,
                 sigma1=0.1, sigma2=0.1, e_cutoff=0.6,
                 alpha1=1, alpha2=1, e_sigma=0.01,
                 reshape=False, e_method='normal',
                 use_intercept=True):
        super().__init__(k, d, patient_id, barriers, prev)
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.e_cutoff = e_cutoff
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.reshape= reshape
        self.c_ij_stars = []  # Logged in or not
        self.received_msgs = []
        self.rewards = []
        self.adherences = []
        self.engagements = []
        self._adherence = 0
        self.e_sigma = e_sigma
        self._engagement = 0.8
        self.prev_engagement = self._engagement  # initial
        self.e_method = e_method
        self.e_msg_prev2_day = 0

    def save_patient_data(self):
        df = pd.DataFrame({'received_msgs': self.received_msgs[:400],
                           'rewards': self.rewards[:400],
                           'logins': self.c_ij_stars[:400],
                           'adherence': self.adherences[:400],
                           'engagements': self.engagements[:400]})
        df.to_csv('./results/{}-{}.csv'.format(self.patient_id,
            datetime.strftime(datetime.now(), '%m%d%H%M%S')))

    def _set_alphas(self):
        return np.array([barrier * np.random.normal(0.65, 0.01)
                         if barrier == 1 else 1 for barrier in self.barriers])

    def engagement(self, prev_engagement):
        self.e_msg_prev1_day = (True if len(self.received_msgs) > 0 and
                                self.received_msgs[-1] == 3 else False)
        self.e_msg_prev2_day = (True if len(self.received_msgs) > 1 and
                                self.received_msgs[-2] == 3 else False)
        self.e_msg_prev3_day = (True if len(self.received_msgs) > 2 and
                                self.received_msgs[-3] == 3 else False)

        if self.e_method =='normal':
            e_i = np.random.normal(loc=0, scale=self.sigma1)
            e_ij = np.random.normal(loc=0, scale=self.sigma2)
            prolong_effect = 1
            e = np.exp(e_i + e_ij +
                       self.alpha1 * self.e_msg_prev1_day +
                       self.alpha2 * self.e_msg_prev2_day +
                       prolong_effect * self.e_msg_prev3_day
                       ) / (1 + np.exp(e_i + e_ij))
            self.engagements.append(e)
            #print('engagement called')
            return e
        elif self.e_method =='ar':
            if prev_engagement > self.e_cutoff:
                e = 0.9*prev_engagement + np.random.normal(0, self.e_sigma) + 30 * self.alpha1 * self.e_msg_prev1_day
            else:
                e = np.random.normal(self.e_cutoff, 0.1)
            self.engagements.append(e)
            return e

    def adherence(self, action):
        #print('adherence called')
        self._engagement = self.engagement(self.prev_engagement)
        self.prev_engagement = self._engagement
        if self._engagement < self.e_cutoff:
            self.c_ij_stars.append(0)
            adherence = 0.02 if action == 3 else 0.01
            self.adherences.append(adherence)
            return adherence
        else:
            self.c_ij_stars.append(1)
            adherence = 0.02 if action == 3 else np.prod(self.alphas, axis=0)
            self.adherences.append(adherence)
            return adherence

    @property
    def c_ij(self):
        self.login_days = np.where(self.c_ij_stars)[0]  # index of elements > 0
        if len(self.login_days) > 1:
            c_ij = self.login_days[-1] - self.login_days[-2] - 1
        elif len(self.login_days) == 1:
            c_ij = self.login_days[-1]
        else:
            c_ij = 0

        return c_ij

    def additional_reward(self):
        f = self.prob_reward1_given(self.c_ij) - self.prob_reward1_given(self.c_ij - 1)
        #print("Additional reward: {}".format(f))
        return f

    def prob_reward1_given(self, c_ij):
        if c_ij < 0:
            return 0
        mask = []
        for idx, login_day in enumerate(self.login_days):
            if idx == 0:
                if login_day == c_ij:
                    mask.append(login_day)
            elif self.login_days[idx] - self.login_days[idx-1] - 1 == c_ij:
                mask.append(login_day)
        #print('mask: {}'.format(mask))
        #print('rewards: {}'.format(self.rewards))
        #print('login_days: {}'.format(self.login_days))  # 확인해보기...
        #print('c_ij_stars: {}'.format(self.c_ij_stars))
        #print('engagements: {}'.format(self.engagements))
        rewards_given_c = np.array(self.rewards)[mask]

        return (sum(rewards_given_c[-5:]) / len(rewards_given_c[-5:])
                if len(rewards_given_c) > 0 else 0)

    def _update_state(self, action, iseffective):
        if self.memory[action].size:
            self.memory[action] = np.vstack((self.memory[action],
                                             np.array([int(iseffective)])))
        else:
            self.memory[action] = np.array([int(iseffective)])

        if self.use_intercept:
            self.states[action] = np.array([1,
                                            sum(self.memory[action][-self.prev:]),
                                            self.e_msg_prev2_day])
        else:
            self.states[action] = np.array([sum(self.memory[action][-self.prev:]),
                                            self.e_msg_prev2_day])
                                            # self.e_msg_prev2_day])  # FIXME

    def _update_adherence(self, match):
        self.alphas = self._set_alphas()

        better_rewards = True
        if match == 3:
            pass
        elif self.e_msg_prev2_day and better_rewards:  # experimental
            beta = 0.5 * np.random.normal(loc=0.7, scale=0.01)
            self.alphas[match] += beta * (1.0 - self.alphas[match])
        elif self.barriers[match]:
            beta = np.random.normal(loc=0.7, scale=0.01)  # loc=0.7, scale=0.3
            self.alphas[match] += beta * (1.0 - self.alphas[match])

    def pull(self, action):
        self._update_adherence(action)
        self.optimal = np.where((self.alphas < 1 ) & (self.barriers == 1))[0]

        #print('pulled: {}'.format(self))
        self._adherence = self.adherence(action)
        if np.random.rand() < self._adherence:
            self._update_state(action, True)
            reward = 1
            self.rewards.append(reward)
            if self.reshape:
                reward = 1 + self.additional_reward()
                self.rewards[-1] = reward
            self.received_msgs.append(action)
            return reward, action in self.optimal
        else:
            self._update_state(action, False)
            reward = 0
            self.rewards.append(reward)
            self.received_msgs.append(action)
            #print("rewards: {}".format(self.rewards))
            return reward, action in self.optimal
