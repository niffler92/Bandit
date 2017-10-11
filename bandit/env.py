import sys
sys.path.append("../")

from bandit.bandit import MultiBandits


class MultiBanditEnv(object):
    def __init__(self, iterations, epochs):
        self.iterations = iterations
        self.epochs = epochs
        self._rewards = dict()

    @property
    def rewards(self):
        """Can be customized
        """
        return pd.DataFrame(self._rewards)

    def make_env(self, bandits):
        """Custom environment for Multibandits
        Args:
            bandits - list of bandit objects
        """
        self.m_bandits = MultiBandits()
        for bandit in bandits:
            self.m_bandits.add_bandit(bandit)

    def reset_env(self):
        for bandit in self.m_bandits.bandits:
            bandit.reset()


    def run(self, agent):
        raise NotImplementedError
