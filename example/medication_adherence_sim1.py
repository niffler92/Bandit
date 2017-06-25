import numpy as np
import matplotlib.pyplot as plt

from bandit.agent import ContextualAgent
from bandit.bandit import MultiBandits
from bandit.policy import LinUCBPolicy, RandomPolicy
from example.sim1_bandit import Patient


def main():
    patients = np.vstack(
                    (np.tile([1, 0, 0], (20,1)),
                     np.tile([0, 1, 0], (20,1)),
                     np.tile([1, 0, 1], (20,1))))

    m_bandits = MultiBandits()
    for patient_id, barriers in enumerate(patients):
        patient = Patient(3, 1, barriers=barriers, patient_id=patient_id)
        m_bandits.add_bandit(patient)
    np.random.shuffle(m_bandits.bandits)

    alpha = 2  # parameter
    d = 1
    episodes = 400
    initial_exploration = 10

    linucb = LinUCBPolicy(alpha, d)
    # randpolicy = RandomPolicy()
    c_agent = ContextualAgent(m_bandits.k, d, linucb)

    average_adh = list()
    average_adh.append(np.mean([bandit.adherence for bandit in m_bandits.bandits]))

    for episode in range(episodes):

        for _ in range(len(m_bandits.bandits)):
            m_bandits.get_bandit()
            c_agent.get_state(m_bandits.bandit)
    ########RANDOM#######################
            # action = np.random.randint(c_agent.k)
            # c_agent.last_action = action
    #####################################

    ########Tailored#####################
            # if _ % 3 == 0:
            #     action = np.random.randint(c_agent.k)
            #     c_agent.last_action = action
            # else:
            #     action = np.random.choice(np.where(m_bandits.bandit.barriers==1)[0])
            #     c_agent.last_action = action
    ####################################

    #########RL############
            action = c_agent.choose(random_period=len(m_bandits.bandits)*initial_exploration)
    ########################
            reward = m_bandits.pull(action)
            c_agent.observe(reward[0])

        average_adh.append(np.mean([bandit.adherence for bandit in m_bandits.bandits]))
    plt.plot(average_adh)
    plt.title("Average adherence rate of patients")
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    plt.xlabel("Last 50 mean adherence: {}".format(np.mean(average_adh[-50:])))
    plt.show()


if __name__ == '__main__':
    main()
