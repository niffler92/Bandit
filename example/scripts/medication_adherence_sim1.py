import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bandit.agent import ContextualAgent
from bandit.bandit import MultiBandits
from bandit.policy import LinUCBPolicy, RandomPolicy
from example.sim1_bandit_intercept import Patient
# from example.sim1_bandit import Patient


def main():
    patients = np.vstack(
                    (np.tile([1, 0, 0], (20, 1)),
                     np.tile([0, 1, 0], (20, 1)),
                     np.tile([1, 0, 1], (20, 1))))

    methods = ['random', 'tailored', 'rl-5', 'rl-7', 'rl-10']

    adh = dict()
    for method in methods:

        alpha = 0.1  # parameter
        k = 3
        d = 2
        iterations = 400
        initial_exploration = 10

        m_bandits = MultiBandits()
        for patient_id, barriers in enumerate(patients):
            if method == 'rl-5':
                patient = Patient(k, d, barriers=barriers, patient_id=patient_id, prev=5)
                m_bandits.add_bandit(patient)
            elif method == 'rl-7':
                patient = Patient(k, d, barriers=barriers, patient_id=patient_id, prev=7)
                m_bandits.add_bandit(patient)
            elif method == 'rl-10':
                patient = Patient(k, d, barriers=barriers, patient_id=patient_id, prev=10)
                m_bandits.add_bandit(patient)
            else:
                patient = Patient(k, d, barriers=barriers, patient_id=patient_id)
                m_bandits.add_bandit(patient)
        # np.random.shuffle(m_bandits.bandits)

        linucb = LinUCBPolicy(alpha, d)
        # randpolicy = RandomPolicy()
        c_agent = ContextualAgent(m_bandits.k, d, linucb)

        average_adh = list()
        average_adh.append(np.mean([bandit.adherence for bandit in m_bandits.bandits]))

        for iteration in range(iterations):
            for _ in range(len(m_bandits.bandits)):
                m_bandits.get_bandit()
                c_agent.get_state(m_bandits.bandit)
                if method == 'random':
                    action = np.random.randint(c_agent.k)
                    c_agent.last_action = action

                if method == 'tailored':
                    # if _ % 3 == 0:
                    #     action = np.random.randint(c_agent.k)
                    #     c_agent.last_action = action
                    if 20 <= m_bandits.bandit.patient_id < 40:
                        action = 2
                        c_agent.last_action = action
                    else:
                        action = np.random.choice(np.where(m_bandits.bandit.barriers == 1)[0])
                        c_agent.last_action = action

                if 'rl' in method:
                    action = c_agent.choose(random_period=len(m_bandits.bandits)*initial_exploration)

                reward = m_bandits.pull(action)
                c_agent.observe(reward[0])

            average_adh.append(np.mean([bandit.adherence for bandit in m_bandits.bandits]))

        adh[method] = average_adh

    df = pd.DataFrame(adh)

    return df


if __name__ == '__main__':
    episodes = 1
    result = 0
    for _ in range(episodes):
        result += main()
        print(_)

    result /= episodes
    ax = result.plot(title="Average adherence rate of patients", legend=True,
                 yticks=[0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set(xlabel='alpha = {}'.format(0.1))
    plt.show()
