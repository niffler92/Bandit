import os
import sys
from datetime import datetime
sys.path.append("../../")

import numpy as np
import pandas as pd

from bandit.env import MultiBanditEnv

class PatientEnv(MultiBanditEnv):
    def __init__(self, iterations, epochs):
        super().__init__(iterations, epochs)
        self._average_adhs = dict()
        self.result = 0
        # this is not an observable data in the real world

    @property
    def average_adhs(self):
        return pd.DataFrame(self._average_adhs)

    @property
    def rewards(self):
        return pd.DataFrame(self._rewards)

    def _reset_per_epoch(self, agent):
        agent.reset()
        self._average_adhs = dict()

    def run(self, agent, tailored=True):
        for _ in range(self.epochs):
            self._run_epoch(agent, tailored)
            self.result += self.average_adhs
            self.average_adhs_once = self.average_adhs
            self._reset_per_epoch(agent)
        self.result = self.result / self.epochs
        self._average_adhs = self.result

    def _run_epoch(self, agent, tailored):
        if tailored:
            policies = ['rl', 'random', 'tailored']
        else:
            policies = ['rl', 'random']

        for policy in policies:
            self.reset_env()
            self._rewards[policy] = []
            average_adh = []
            average_adh.append(np.mean([
                bandit.adherence for bandit in self.m_bandits.bandits]))
            for iteration in range(self.iterations):
                for _ in range(len(self.m_bandits.bandits)):
                    self.m_bandits.get_bandit()
                    agent.get_state(self.m_bandits.bandit)
                    # Only contextual bandit has method get_state
                    action = self._get_action(agent, policy, iteration)
                    agent.last_action = action

                    reward = self.m_bandits.pull(action)[0]
                    self._rewards[policy].append(reward)
                    agent.observe(reward)

                average_adh.append(np.mean(
                    [bandit.adherence for bandit in self.m_bandits.bandits]))
                # originally bnadit._adherence

            self._average_adhs[policy] = average_adh

    def _get_action(self, agent, policy, iteration):
        if policy == 'rl':
            return agent.choose()
        elif policy == 'random':
            return np.random.randint(agent.k)
        elif policy == 'tailored':
            if 20 <= self.m_bandits.bandit.patient_id < 40:
                action = 2
            else:
                action = np.random.choice(
                    np.where(self.m_bandits.bandit.barriers == 1)[0])
            return action

    def plot_adherence(self, alpha):
        ax = self.result.plot(title="Average adherence rate of patients",
                              legend=True,
                              yticks=[0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set(xlabel = "alpha = {}".format(alpha))
        plt.show()

    def save_result(self, alpha, epochs, filename, save_path='./'):
        #self.result.to_csv(
        #    os.path.join(save_path,
        #                 filename + datetime.strftime(datetime.now(),
        #                                              '%Y%m%d-%H%M.csv'))
        #)

        ax = self.result.plot(title="Average adherence rate of patients",
                              legend=True,
                              yticks=[0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set(xlabel = "alpha = {}, epochs = {}".format(alpha, epochs))
        fig = ax.get_figure()
        fig.savefig(
            os.path.join(save_path,
                         filename + datetime.strftime(datetime.now(),
                                                      'plot_%Y%m%d-%H%M.png'))
        )


class PatientEnv2(PatientEnv):
    def __init__(self, iterations, epochs):
        super().__init__(iterations, epochs)

    def _get_action(self, agent, policy, iteration):
        if policy == 'rl':
            return agent.choose()
        elif policy == 'random':
            return np.random.randint(agent.k)
        elif policy == 'tailored':
            if episode >= episodes / 2 and 20 <= m_bandits.bandit.patient_id < 40:
                action = 1
            else:
                action = np.random.choice(np.where(m_bandits.bandit.barriers == 1)[0])
            return action
