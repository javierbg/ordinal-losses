import torch
from torch import nn
import torch.nn.functional as F
import warnings

############################## UTILITIES #####################################

def fact(x):
    return torch.exp(torch.lgamma(x+1))

def log_fact(x):
    return torch.lgamma(x+1)

# we are using softplus instead of relu since it is smoother to optimize.
# as in http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf
approx_relu = F.softplus
ce = nn.CrossEntropyLoss()

################################ LOSSES ######################################

class CrossEntropy:
    def __init__(self, K):
        self.K = K

    def how_many_outputs(self):
        # how many output neurons does this loss require?
        return self.K

    def activation(self, Yhat):
        # output post-processing, if necessary
        return Yhat

    def __call__(self, Yhat, Y):
        # computes the loss
        return ce(self.activation(Yhat), Y)

    def to_proba(self, Yhat):
        # call output -> probabilities
        return F.softmax(self.activation(Yhat), 1)

    def to_classes(self, Phat, method=None):
        # None=default; this is typically 'mode', but can be different for each
        # loss.
        assert method in (None, 'mode', 'mean', 'median')
        if method in (None, 'mode'):
            return Phat.argmax(1)
        if method == 'mean':  # so-called expectation trick
            kk = torch.arange(args.classes, device=Phat.device)
            return torch.round(torch.sum(Yhat * kk, 1)).long()
        if method == 'median':
            # the weighted median is the value whose cumulative probability is 0.5
            Pc = torch.cumsum(Phat, 1)
            return torch.sum(Pc < 0.5, 1)

    def to_proba_and_classes(self, Yhat, method=None):
        Phat = self.to_proba(Yhat)
        Khat = self.to_classes(Phat, method)
        return Phat, Khat

class MAE:
    def __call__(self, Yhat, Y):
        Phat = torch.softmax(Yhat, 1)
        return F.l1_loss(Phat, Y)

class MSE:
    def __call__(self, Yhat, Y):
        Phat = torch.softmax(Yhat, 1)
        return F.mse_loss(Phat, Y)

##############################################################################
# Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network      #
# approach to ordinal regression." 2008 IEEE international joint conference  #
# on neural networks (IEEE world congress on computational intelligence).    #
# IEEE, 2008. https://arxiv.org/pdf/0704.1028.pdf                            #
##############################################################################

class OrdinalEncoding(CrossEntropy):
    def how_many_outputs(self):
        return self.K-1

    def __call__(self, Yhat, Y):
        # if K=4, then
        #                k = 0  1  2
        #     Y=0 => P(Y>k)=[0, 0, 0]
        #     Y=1 => P(Y>k)=[1, 0, 0]
        #     Y=2 => P(Y>k)=[1, 1, 0]
        #     Y=3 => P(Y>k)=[1, 1, 1]
        KK = torch.arange(self.K-1, device=Y.device).expand(Y.shape[0], -1)
        YY = (Y[:, None] > KK).float()
        return F.binary_cross_entropy_with_logits(Yhat, YY)

    def to_proba(self, Yhat):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y>k) into P(Y=k)
        # P(Y=0) = 1-P(Y>0)
        # P(Y=1) = P(Y>0)-P(Y>1)
        # ...
        # P(Y=K-1) = P(Y>K-2)
        Phat = torch.sigmoid(Yhat)
        Phat = torch.cat((1-Phat[:, :1], Phat[:, :-1]-Phat[:, 1:], Phat[:, -1:]), 1)
        # there may be small discrepancies
        Phat = torch.clamp(Phat, 0, 1)
        Phat = Phat / Phat.sum(1, keepdim=True)
        return Phat

    def to_classes(self, Phat, method=None):
        warnings.warn('OrdinalEncoding.to_classes(): To use the same algorithm as the paper, use to_proba_and_classes(output) instead of to_classes(to_proba(output)) separately.')
        return super().to_classes(Phat, method)

    def to_proba_and_classes(self, Yhat, method=None):
        if method is None:
            Phat = self.to_proba(Yhat)
            Khat = torch.sum(Yhat >= 0, 1)
            return Phat, Khat
        return super().to_proba_and_classes(self, Yhat, method)

##############################################################################
# da Costa, Joaquim F. Pinto, Hugo Alonso, and Jaime S. Cardoso. "The        #
# unimodal model for the classification of ordinal data." Neural Networks    #
# 21.1 (2008): 78-91.                                                        #
# https://www.sciencedirect.com/science/article/pii/S089360800700202X        #
##############################################################################

class BinomialUnimodal_CE(CrossEntropy):
    def how_many_outputs(self):
        return 1

    def __call__(self, Yhat, Y):
        return F.nll_loss(self.to_log_proba(Yhat), Y)

    def to_proba(self, Yhat):
        device = Yhat.device
        Phat = torch.sigmoid(Yhat)
        N = Yhat.shape[0]
        K = torch.tensor(self.K, dtype=torch.float, device=device)
        kk = torch.ones((N, self.K), device=device) * torch.arange(self.K, dtype=torch.float, device=device)[None]
        num = fact(K-1) * (Phat**kk) * (1-Phat)**(K-kk-1)
        den = fact(kk) * fact(K-kk-1)
        return num / den

    def to_log_proba(self, Yhat):
        device = Yhat.device
        log_Phat = F.logsigmoid(Yhat)
        log_inv_Phat = F.logsigmoid(-Yhat)
        N = Yhat.shape[0]
        K = torch.tensor(self.K, dtype=torch.float, device=device)
        kk = torch.ones((N, self.K), device=device) * torch.arange(self.K, dtype=torch.float, device=device)[None]
        num = log_fact(K-1) + kk*log_Phat + (K-kk-1)*log_inv_Phat
        den = log_fact(kk) + log_fact(K-kk-1)
        return num - den

class BinomialUnimodal_MSE(BinomialUnimodal_CE):
    def __call__(self, Yhat, Y):
        device = Yhat.device
        Phat = self.to_proba(Yhat)
        Y_onehot = torch.zeros(Phat.shape[0], self.K, device=device)
        Y_onehot[range(Phat.shape[0]), Y] = 1
        return torch.mean((Phat - Y_onehot) ** 2)

##############################################################################
# Beckham, Christopher, and Christopher Pal. "Unimodal probability           #
# distributions for deep ordinal classification." International Conference   #
# on Machine Learning. PMLR, 2017.                                           #
# http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf                 #
##############################################################################

class PoissonUnimodal(CrossEntropy):
    def how_many_outputs(self):
        return 1

    def activation(self, Yhat):
        # they apply softplus (relu) to avoid log(negative)
        Yhat = F.softplus(Yhat)
        KK = torch.arange(1., self.K+1, device=Yhat.device)[None]
        return KK*torch.log(Yhat) - Yhat - log_fact(KK)

##############################################################################
# de La Torre, Jordi, Domenec Puig, and Aida Valls. "Weighted kappa loss     #
# function for multi-class classification of ordinal data in deep learning." #
# Pattern Recognition Letters 105 (2018): 144-154.                           #
# https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666    #
##############################################################################
# Use n=2 (default) for Quadratic Weighted Kappa.                            #
##############################################################################

class WeightedKappa(CrossEntropy):
    def __init__(self, K, n=2):
        self.K = K
        self.n = 2

    def __call__(self, Yhat, Y):
        Phat = torch.softmax(Yhat, 1)
        kk = torch.arange(self.K, device=Y.device)
        i, j = torch.meshgrid(kk, kk, indexing='xy')
        w = torch.abs(i-j)**self.n
        N = torch.sum(w[Y] * Phat)
        Phat_sum = torch.sum(Phat, 0)
        D = sum((torch.sum(Y == i)/len(Y)) * torch.sum(w[i] * Phat_sum) for i in range(self.K))
        kappa = 1 - N/D
        return torch.log(1-kappa)

##############################################################################
# Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Ordinal losses for #
# for classification of cervical cancer risk." PeerJ Computer Science 7      #
# (2021): e457. https://peerj.com/articles/cs-457/                           #
##############################################################################
# These losses require two parameters: omega and lambda.                     #
# The default omega value comes from the paper.                              #
# The default lambda values comes from our experiments.                      #
##############################################################################

def entropy_term(Yhat):
    # https://en.wikipedia.org/wiki/Entropy_(information_theory)
    P = F.softmax(Yhat, 1)
    logP = F.log_softmax(Yhat, 1)
    N = P.shape[0]
    return -torch.sum(P * logP) / N

def neighbor_term(Yhat, Y, margin):
    margin = torch.tensor(margin, device=Y.device)
    P = F.softmax(Yhat, 1)
    K = P.shape[1]
    dP = torch.diff(P, 1)
    sign = (torch.arange(K-1, device=Y.device)[None] >= Y[:, None])*2-1
    return torch.mean(torch.sum(approx_relu(margin + sign*dP, 1)))

class CO2(CrossEntropy):
    def __init__(self, K, lamda=0.01, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def __call__(self, Yhat, Y):
        return ce(Yhat, Y) + self.lamda*neighbor_term(Yhat, Y, self.omega)

class CO(CO2):
    # CO is the same as CO2 with omega=0
    def __init__(self, K, lamda=0.01, omega=0.05):
        super().__init__(K, lamda, 0)

class HO2(CrossEntropy):
    def __init__(self, K, lamda=1.0, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def __call__(self, Yhat, Y):
        return entropy_term(Yhat) + self.lamda*neighbor_term(Yhat, Y, self.omega)

##############################################################################
# Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Quasi-Unimodal     #
# Distributions for Ordinal Classification." Mathematics 10.6 (2022): 980.   #
# https://www.mdpi.com/2227-7390/10/6/980                                    #
##############################################################################
# These losses require two parameters: omega and lambda.                     #
# The default omega value comes from the paper.                              #
# The default lambda values comes from our experiments.                      #
##############################################################################

def quasi_neighbor_term(Yhat, Y, margin):
    margin = torch.tensor(margin, device=Y.device)
    P = F.softmax(Yhat, 1)
    K = P.shape[1]
    ix = torch.arange(len(P))

    # force close neighborhoods to be inferior to True class prob
    has_left = Y > 0
    close_left = has_left * approx_relu(margin+P[ix, Y-1]-P[:, Y])
    has_right = Y < K-1
    close_right = has_right * approx_relu(margin+P[ix, (Y+1)%K]-P[:, Y])

    # force distant probabilities to be inferior than close neighborhoods of true class
    left = torch.arange(K, device=Y.device)[None] < Y[:, None]-1
    distant_left = torch.sum(left * approx_relu(margin+P-P[ix, Y-1][:, None]), 1)
    right = torch.arange(K, device=Y.device)[None] > Y[:, None]+1
    distant_right = torch.sum(right * approx_relu(margin+P-P[ix, (Y+1)%K][:, None]), 1)

    return torch.mean(close_left + close_right + distant_left + distant_right)

class QUL_CE(CrossEntropy):
    def __init__(self, K, lamda=0.1, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def __call__(self, Yhat, Y):
        return ce(Yhat, Y) + self.lamda*quasi_neighbor_term(Yhat, Y, self.omega)

class QUL_HO(CrossEntropy):
    def __init__(self, K, lamda=10., omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def __call__(self, Yhat, Y):
        return entropy_term(Yhat) + self.lamda*quasi_neighbor_term(Yhat, Y, self.omega)


##############################################################################
# Polat, Gorkem, et al. "Class Distance Weighted Cross-Entropy Loss for      #
# Ulcerative Colitis Severity Estimation." arXiv preprint arXiv:2202.05167   #
# (2022). https://arxiv.org/pdf/2202.05167.pdf                               #
##############################################################################

class CDW_CE(CrossEntropy):
    # Reference: https://arxiv.org/pdf/2202.05167.pdf
    def __init__(self, K, alpha=5):
        super().__init__(K)
        self.alpha = alpha

    def __call__(self, Yhat, Y):
        Yhat = F.softmax(Yhat, 1)
        return -torch.mean(torch.log(1-Yhat) * torch.abs(torch.arange(self.K, device=Y.device)[None]-Y[:, None])**self.alpha)

##############################################################################
# To be published.                                                           #
##############################################################################

class UnimodalNet(CrossEntropy):
    def activation(self, Yhat):
        # first use relu: we need everything positive
        # for differentiable reasons, we use leaky relu
        Yhat = approx_relu(Yhat)
        # if output=[X,Y,Z] => pos_slope=[X,X+Y,X+Y+Z]
        # if output=[X,Y,Z] => neg_slope=[Z,Z+Y,Z+Y+X]
        pos_slope = torch.cumsum(Yhat, 1)
        neg_slope = torch.flip(torch.cumsum(torch.flip(Yhat, [1]), 1), [1])
        Yhat = torch.minimum(pos_slope, neg_slope)
        return Yhat

def unimodal_wasserstein(p, mode):
    # Returns the closest unimodal distribution to p with the given mode.
    # Return tuple:
    # 0: total transport cost
    # 1: closest unimodal distribution
    import numpy as np
    from scipy.spatial.distance import squareform, pdist
    from scipy.optimize import linprog
    assert abs(p.sum()-1) < 1e-6, 'Expected normalized probability mass.'
    assert np.any(p >= 0), 'Expected nonnegative probabilities.'
    assert len(p.shape) == 1, 'Probabilities p must be a vector.'
    assert 0 <= mode < p.size, 'Invalid mode value.'
    K = p.size
    C = squareform(pdist(np.arange(K)[:, None]))  # cost matrix
    Ap = [([0]*i + [1] + [0]*(K-i-1))*K for i in range(K)]
    Ai = [[0]*i*K + [1]*K + [-1]*K + [0]*(K-i-2)*K if i < mode else
          [0]*i*K + [-1]*K + [1]*K + [0]*(K-i-2)*K for i in range(K-1)]
    result = linprog(C.ravel(), A_ub=Ai, b_ub=np.zeros(K-1), A_eq=Ap, b_eq=p)
    T = result.x.reshape(K, K)
    return (T*C).sum(), T.sum(1)

def emd(p, q):
    # https://en.wikipedia.org/wiki/Earth_mover%27s_distance
    pp = p.cumsum(1)
    qq = q.cumsum(1)
    return torch.mean(torch.sum(torch.abs(pp-qq), 1))

def is_unimodal(p):
    # checks (true/false) whether the given probability vector is unimodal. this
    # function is not used by the following classes, but it is used in the paper
    # to compute the "% times unimodal" metric
    zero = torch.zeros(1, device=p.device)
    p = torch.sign(torch.round(torch.diff(p, prepend=zero, append=zero), decimals=2))
    p = torch.diff(p[p != 0])
    p = p[p != 0]
    return len(p) <= 1

class WassersteinUnimodal_KLDIV(CrossEntropy):
    def __init__(self, K, lamda=100.):
        super().__init__(K)
        self.lamda = lamda

    def distance_loss(self, phat, phat_log, target):
        return F.kl_div(phat_log, target, reduction='batchmean')

    def __call__(self, Yhat, Y):
        Phat = torch.softmax(Yhat, 1)
        Phat_log = F.log_softmax(Yhat, 1)
        closest_unimodal = torch.stack([
            torch.tensor(unimodal_wasserstein(phat, y)[1], dtype=torch.float32, device=Y.device)
            for phat, y in zip(Phat.cpu().detach().numpy(), Y.cpu().numpy())])
        return ce(Yhat, Y) + self.lamda*self.distance_loss(Phat, Phat_log, closest_unimodal)

class WassersteinUnimodal_EMD(WassersteinUnimodal_KLDIV):
    def distance_loss(self, phat, phat_log, target):
        return emd(phat, target)
