import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bandit.agent import ContextualAgent
from bandit.bandit import MultiBandits
from bandit.policy import LinUCBPolicy, RandomPolicy
from example.sim3_bandit import Patient


def main():
    patients = np.vstack(
                    (np.tile([1, 0, 0], (20, 1)),
                     np.tile([0, 1, 0], (20, 1)),
                     np.tile([1, 0, 1], (20, 1))))
    methods = ['random', 'tailored', 'rl']
    # methods = ['rl']
    adh = dict()
    for method in methods:

        alpha = 1  # parameter
        k = 4
        d = 3
        iterations = 400
        initial_exploration = 10

        m_bandits = MultiBandits()
        for patient_id, barriers in enumerate(patients):
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
                    action = np.random.randint(c_agent.k-1)
                    c_agent.last_action = action

                if method == 'tailored':
                    action = np.random.choice(np.where(m_bandits.bandit.barriers == 1)[0])
                    c_agent.last_action = action

                if method == 'rl':
                    action = c_agent.choose(random_period=len(m_bandits.bandits)*initial_exploration)

                reward = m_bandits.pull(action)
                # if method == 'rl':
                #     if action == 3:
                #         if m_bandits.bandit.isLingering:
                #             print("Now Lingering, {}, fatigue = {}".format(m_bandits.bandit,
                #                                                            m_bandits.bandit.fatigue_fct))
                #             print("Adherence: {}".format(m_bandits.bandit.adherence))
                #             print("Alphas: {}".format(m_bandits.bandit.alphas))
                #             print(m_bandits.bandit.msg_sent)
                #             # print("Best observation !!!!!!!!!!!!!")
                #     if m_bandits.bandit.patient_id == 1:
                #         print('Fatigue of patient1:{}, adherence:{}'.format(m_bandits.bandit.fatigue_fct,
                #                                                             m_bandits.bandit.adherence))
                #         print('When no message is sent, context:', m_bandits.bandit.states[3])
                #         print('Msg sent : {}'.format(m_bandits.bandit.msg_sent))
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
    ax.set(xlabel='alpha = {}'.format(1))
    plt.show()
