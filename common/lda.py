import re
import numpy as np
from scipy.special import gammaln, psi

np.random.seed(1337)

def parse_string_list(l):
    """
    Ex.:
    Input: ["Hello World, Hello you", "what?"] 
    Returns: [{'hello': 2, 'world': 1, 'you': 1}, {'what': 1}]

    TODO TEST
    """
    r = []
    for s in l:
        s = s.lower()
        s = s.replace('\n', ' ')
        s = re.sub('[^a-z ]', '')
        tmp = {}
        for w in s.split(' '):
            if w != '' and w != ' ':
                tmp[w] = tmp.get(w, 0) + 1
        r.append(tmp)
    return r

class LDA:
    """
    TODO
    """

    def __init__(self, method='vb', K=1000, D=1000000, alpha=0.01, eta=0.01):
        """
        method = 'vb' or 'gibbs'
        K: (max) number of topic
        D: (max) number of documents
        alpha: hyperparameter on thetas
        eta: hyperparameter on betas

        TODO add smoothing
        """
        self._method = method
        self._K = K
        self._D = D
        self._alpha = alpha
        self._eta = eta

    def __repr__(self):
        print "LDA with", self._method, "inference"
        print "alpha:", self._alpha, "eta:", self._eta
        print "TODO RESULTS"

    def variational(self, docs):
        """
        docs is a list of dictionaries [{'word': count}]

        Approximates p(theta, z | w, alpha, beta) by q(theta, z | gamma, Phi)
        = q(theta | gamma).\prod_{n=1}^N q(z_n | Phi_n)

        Algorithm (Blei, Ng, Jordan: Latent Dirichlet Allocation, JMLR 2003):
          Phi_{n,i}^0 = 1/K for all i and n
          gamma_i = alpha_i + N/K for all i
          until convergence:
            for n=1 to N:
              for i=1 to K:
                Phi_{n,i}^{t+1} = beta_{i,w_n} exp(Psi(gamma_i^t))
              normalize Phi_n^t to sum to 1
            gamma^{t+1} = alpha + \sum_{n=1}^N Phi_n^{t+1}
        with Psi being the first derivative of the logGamma function
        """


    def fit(self, X):
        """
        Compute LDA for X (can be a list of string or a list of dicts)
        """
        assert(len(X) > 0)
        if type(X[0]) == str:
            X = parse_string_list(X)
        assert(type(X[0]) == dict)
        if self._method == 'vb':
            variational(X)
            






         




