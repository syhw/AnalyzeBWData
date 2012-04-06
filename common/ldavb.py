import numpy as np
from scipy.special import gammaln, psi

np.random.seed(1337)

class LDAVB:
    """
    TODO
    """

    def __init__(self, K=1000, D=1000000, alpha=0.01, eta=0.01):
        """
        K: (max) number of topic
        D: (max) number of documents
        alpha: hyperparameter on thetas
        eta: hyperparameter on betas
        """
        self._K = K
        self._D = K
        self._alpha = K
        self._eta = K

    def estimate(self, docs):
         




