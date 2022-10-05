import torch
from torch import nn
import torch.nn.functional as F

############################## UTILITIES #####################################

def fact(x):
    return torch.exp(torch.lgamma(x+1))

def log_fact(x):
    return torch.lgamma(x+1)

def to_classes(probs, method=None):
    # None=default; this is typically 'mode', but can be different for each
    # loss.
    assert method in (None, 'mode', 'mean', 'median')
    if method == 'mean':  # so-called expectation trick
        kk = torch.arange(args.classes, device=probs.device)
        return torch.round(torch.sum(ypred * kk, 1)).long()
    elif method == 'median':
        # the weighted median is the value whose cumulative probability is 0.5
        Pc = torch.cumsum(probs, 1)
        return torch.sum(Pc < 0.5, 1)
    else:  # default=mode
        return probs.argmax(1)

# we are using softplus instead of relu since it is smoother to optimize.
# as in http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf
approx_relu = F.softplus
ce = nn.CrossEntropyLoss(reduction='none')

################################ LOSSES ######################################

class CrossEntropy:
    def __init__(self, K):
        self.K = K

    def set_model(self, model):
        # some losses may use this to install learnable parameters
        pass

    def how_many_outputs(self):
        # how many output neurons does this loss require?
        return self.K

    def __call__(self, ypred, ytrue):
        # computes the loss
        return ce(ypred, ytrue)

    def to_proba(self, ypred):
        # output -> probabilities
        return F.softmax(ypred, 1)

    def to_proba_and_classes(self, ypred, method=None):
        # output -> probabilities & classes
        probs = self.to_proba(ypred)
        classes = to_classes(probs, method)
        return probs, classes

class MAE(CrossEntropy):
    def how_many_outputs(self):
        return 1

    def __call__(self, ypred, ytrue):
        ypred = torch.clamp(ypred, 0, self.K-1)[:, 0]
        return torch.abs(ypred-ytrue)

    def to_proba_and_classes(self, ypred, method=None):
        ypred = torch.clamp(ypred, 0, self.K-1)[:, 0].long()
        probs = torch.nn.functional.one_hot(ypred, self.K)
        return probs, ypred

class MSE(MAE):
    def __call__(self, ypred, ytrue):
        ypred = torch.clamp(ypred, 0, self.K-1)[:, 0]
        return (ypred-ytrue)**2

##############################################################################
# Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network      #
# approach to ordinal regression." 2008 IEEE international joint conference  #
# on neural networks (IEEE world congress on computational intelligence).    #
# IEEE, 2008. https://arxiv.org/pdf/0704.1028.pdf                            #
##############################################################################

class OrdinalEncoding(CrossEntropy):
    def how_many_outputs(self):
        return self.K-1

    def __call__(self, ypred, ytrue):
        # if K=4, then
        #                k = 0  1  2
        #     Y=0 => P(Y>k)=[0, 0, 0]
        #     Y=1 => P(Y>k)=[1, 0, 0]
        #     Y=2 => P(Y>k)=[1, 1, 0]
        #     Y=3 => P(Y>k)=[1, 1, 1]
        KK = torch.arange(self.K-1, device=ytrue.device).expand(ytrue.shape[0], -1)
        yytrue = (ytrue[:, None] > KK).float()
        return torch.sum(F.binary_cross_entropy_with_logits(ypred, yytrue, reduction='none'), 1)

    def to_proba(self, ypred):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y>k) into P(Y=k)
        # P(Y=0) = 1-P(Y>0)
        # P(Y=1) = P(Y>0)-P(Y>1)
        # ...
        # P(Y=K-1) = P(Y>K-2)
        probs = torch.sigmoid(ypred)
        probs = torch.cat((1-probs[:, :1], probs[:, :-1]-probs[:, 1:], probs[:, -1:]), 1)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return probs

    def to_proba_and_classes(self, ypred, method=None):
        probs = self.to_proba(ypred)
        if method is None:
            classes = torch.sum(ypred >= 0, 1)
        else:
            classes = to_classes(probs, method)
        return probs, classes

##############################################################################
# da Costa, Joaquim F. Pinto, Hugo Alonso, and Jaime S. Cardoso. "The        #
# unimodal model for the classification of ordinal data." Neural Networks    #
# 21.1 (2008): 78-91.                                                        #
# https://www.sciencedirect.com/science/article/pii/S089360800700202X        #
##############################################################################

class BinomialUnimodal_CE(CrossEntropy):
    def how_many_outputs(self):
        return 1

    def __call__(self, ypred, ytrue):
        return F.nll_loss(self.to_log_proba(ypred), ytrue, reduction='none')

    def to_proba(self, ypred):
        device = ypred.device
        probs = torch.sigmoid(ypred)
        N = ypred.shape[0]
        K = torch.tensor(self.K, dtype=torch.float, device=device)
        kk = torch.ones((N, self.K), device=device) * torch.arange(self.K, dtype=torch.float, device=device)[None]
        num = fact(K-1) * (probs**kk) * (1-probs)**(K-kk-1)
        den = fact(kk) * fact(K-kk-1)
        return num / den

    def to_log_proba(self, ypred):
        device = ypred.device
        log_probs = F.logsigmoid(ypred)
        log_inv_probs = F.logsigmoid(-ypred)
        N = ypred.shape[0]
        K = torch.tensor(self.K, dtype=torch.float, device=device)
        kk = torch.ones((N, self.K), device=device) * torch.arange(self.K, dtype=torch.float, device=device)[None]
        num = log_fact(K-1) + kk*log_probs + (K-kk-1)*log_inv_probs
        den = log_fact(kk) + log_fact(K-kk-1)
        return num - den

class BinomialUnimodal_MSE(BinomialUnimodal_CE):
    def __call__(self, ypred, ytrue):
        device = ypred.device
        probs = self.to_proba(ypred)
        yonehot = torch.zeros(probs.shape[0], self.K, device=device)
        yonehot[range(probs.shape[0]), ytrue] = 1
        return torch.sum((probs - yonehot)**2, 1)

##############################################################################
# Beckham, Christopher, and Christopher Pal. "Unimodal probability           #
# distributions for deep ordinal classification." International Conference   #
# on Machine Learning. PMLR, 2017.                                           #
# http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf                 #
##############################################################################

class PoissonUnimodal(CrossEntropy):
    def how_many_outputs(self):
        return 1

    def activation(self, ypred):
        # they apply softplus (relu) to avoid log(negative)
        ypred = F.softplus(ypred)
        KK = torch.arange(1., self.K+1, device=ypred.device)[None]
        return KK*torch.log(ypred) - ypred - log_fact(KK)

    def __call__(self, ypred, ytrue):
        return ce(self.activation(ypred), ytrue)

    def to_proba(self, ypred):
        return F.softmax(self.activation(ypred), 1)

##############################################################################
# de La Torre, Jordi, Domenec Puig, and Aida Valls. "Weighted kappa loss     #
# function for multi-class classification of ordinal data in deep learning." #
# Pattern Recognition Letters 105 (2018): 144-154.                           #
# https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666    #
##############################################################################
# Use n=2 (default) for Quadratic Weighted Kappa.                            #
# Notice that the other losses are reduction='none'. But this loss, by its   #
# very nature, always returns a scalar.                                      #
##############################################################################

class WeightedKappa(CrossEntropy):
    def __init__(self, K, n=2):
        super().__init__(K)
        self.n = 2

    def __call__(self, ypred, ytrue):
        probs = torch.softmax(ypred, 1)
        kk = torch.arange(self.K, device=ytrue.device)
        i, j = torch.meshgrid(kk, kk, indexing='xy')
        w = torch.abs(i-j)**self.n
        N = torch.sum(w[ytrue] * probs)
        probs_sum = torch.sum(probs, 0)
        D = sum((torch.sum(ytrue == i)/len(ytrue)) * torch.sum(w[i] * probs_sum) for i in range(self.K))
        kappa = 1 - N/D
        return torch.log(1-kappa+1e-7)

##############################################################################
# Vargas, Victor Manuel, Pedro Antonio Gutiérrez, and César Hervás-Martínez. #
# "Cumulative link models for deep ordinal classification." Neurocomputing   #
# 401 (2020): 48-58.                                                         #
# https://www.sciencedirect.com/science/article/pii/S0925231220303805        #
##############################################################################
# This paper is an extension of POM (McCullagh, 1980).                       #
# Loosely based on the code provided by the authors:                         #
# https://github.com/EthanRosenthal/spacecutter                              #
##############################################################################

class CumulativeLinkLoss(CrossEntropy):
    def how_many_outputs(self):
        return 1

    def __init__(self, K, link_function=torch.sigmoid, init_cutpoints='ordered'):
        super().__init__(K)
        assert init_cutpoints in ('ordered', 'random')
        self.link_function = link_function
        self.init_cutpoints = init_cutpoints

    def set_model(self, model):
        self.model = model
        if hasattr(model, 'cutpoints'):
            return
        ncutpoints = self.K-1
        device = next(model.parameters()).device
        params = {'dtype': torch.float32, 'requires_grad': True, 'device': device}
        if self.init_cutpoints == 'ordered':
            model.cutpoints = torch.arange(ncutpoints, **params) - ncutpoints/2
        else:
            model.cutpoints = torch.rand(ncutpoints, **params).sort()[0]

    def __call__(self, ypred, ytrue):
        probs = self.to_proba(ypred)
        return -torch.log(probs[ytrue, 0]+1e-7)  # cross-entropy

    def to_proba(self, ypred):
        ypred = self.link_function(self.model.cutpoints - ypred)
        link_mat = ypred[:, 1:] - ypred[:, :-1]
        return torch.cat((ypred[:, [0]], link_mat, 1-ypred[:, [-1]]), 1)

##############################################################################
# Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Ordinal losses for #
# for classification of cervical cancer risk." PeerJ Computer Science 7      #
# (2021): e457. https://peerj.com/articles/cs-457/                           #
##############################################################################
# These losses require two parameters: omega and lambda.                     #
# The default omega value comes from the paper.                              #
# The default lambda values comes from our experiments.                      #
##############################################################################

def entropy_term(ypred):
    # https://en.wikipedia.org/wiki/Entropy_(information_theory)
    P = F.softmax(ypred, 1)
    logP = F.log_softmax(ypred, 1)
    return -torch.sum(P * logP, 1)

def neighbor_term(ypred, ytrue, margin):
    margin = torch.tensor(margin, device=ytrue.device)
    P = F.softmax(ypred, 1)
    K = P.shape[1]
    dP = torch.diff(P, 1)
    sign = (torch.arange(K-1, device=ytrue.device)[None] >= ytrue[:, None])*2-1
    return torch.sum(approx_relu(margin + sign*dP, 1))

class CO2(CrossEntropy):
    def __init__(self, K, lamda=0.01, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def __call__(self, ypred, ytrue):
        term = neighbor_term(ypred, ytrue, self.omega)
        return ce(ypred, ytrue) + self.lamda*term

class CO(CO2):
    # CO is the same as CO2 with omega=0
    def __init__(self, K, lamda=0.01, omega=0.05):
        super().__init__(K, lamda, 0)

class HO2(CrossEntropy):
    def __init__(self, K, lamda=1.0, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def __call__(self, ypred, ytrue, reduction='mean'):
        term = neighbor_term(ypred, ytrue, self.omega)
        return entropy_term(ypred) + self.lamda*term

##############################################################################
# Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Quasi-Unimodal     #
# Distributions for Ordinal Classification." Mathematics 10.6 (2022): 980.   #
# https://www.mdpi.com/2227-7390/10/6/980                                    #
##############################################################################
# These losses require two parameters: omega and lambda.                     #
# The default omega value comes from the paper.                              #
# The default lambda values comes from our experiments.                      #
##############################################################################

def quasi_neighbor_term(ypred, ytrue, margin):
    margin = torch.tensor(margin, device=ytrue.device)
    P = F.softmax(ypred, 1)
    K = P.shape[1]
    ix = torch.arange(len(P))

    # force close neighborhoods to be inferior to True class prob
    has_left = ytrue > 0
    close_left = has_left * approx_relu(margin+P[ix, ytrue-1]-P[ix, ytrue])
    has_right = ytrue < K-1
    close_right = has_right * approx_relu(margin+P[ix, (ytrue+1)%K]-P[ix, ytrue])

    # force distant probabilities to be inferior than close neighborhoods of true class
    left = torch.arange(K, device=ytrue.device)[None] < ytrue[:, None]-1
    distant_left = torch.sum(left * approx_relu(margin+P-P[ix, ytrue-1][:, None]), 1)
    right = torch.arange(K, device=ytrue.device)[None] > ytrue[:, None]+1
    distant_right = torch.sum(right * approx_relu(margin+P-P[ix, (ytrue+1)%K][:, None]), 1)

    return close_left + close_right + distant_left + distant_right

class QUL_CE(CrossEntropy):
    def __init__(self, K, lamda=0.1, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def __call__(self, ypred, ytrue):
        term = quasi_neighbor_term(ypred, ytrue, self.omega)
        return ce(ypred, ytrue) + self.lamda*term

class QUL_HO(CrossEntropy):
    def __init__(self, K, lamda=10., omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def __call__(self, ypred, ytrue):
        term = quasi_neighbor_term(ypred, ytrue, self.omega)
        return entropy_term(ypred) + self.lamda*term


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

    def __call__(self, ypred, ytrue):
        ypred = F.softmax(ypred, 1)
        return -torch.sum(torch.log(1-ypred) * torch.abs(torch.arange(self.K, device=ytrue.device)[None]-ytrue[:, None])**self.alpha, 1)

##############################################################################
# To be published.                                                           #
##############################################################################

class UnimodalNet(CrossEntropy):
    def activation(self, ypred):
        # first use relu: we need everything positive
        # for differentiable reasons, we use leaky relu
        ypred = approx_relu(ypred)
        # if output=[X,Y,Z] => pos_slope=[X,X+Y,X+Y+Z]
        # if output=[X,Y,Z] => neg_slope=[Z,Z+Y,Z+Y+X]
        pos_slope = torch.cumsum(ypred, 1)
        neg_slope = torch.flip(torch.cumsum(torch.flip(ypred, [1]), 1), [1])
        ypred = torch.minimum(pos_slope, neg_slope)
        return ypred

    def __call__(self, ypred, ytrue):
        return ce(self.activation(ypred), ytrue)

    def to_proba(self, ypred):
        return F.softmax(self.activation(ypred), 1)

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
    return torch.sum(torch.abs(pp-qq), 1)

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
        return torch.sum(F.kl_div(phat_log, target, reduction='none'), 1)

    def __call__(self, ypred, ytrue):
        probs = torch.softmax(ypred, 1)
        probs_log = F.log_softmax(ypred, 1)
        closest_unimodal = torch.stack([
            torch.tensor(unimodal_wasserstein(phat, y)[1], dtype=torch.float32, device=ytrue.device)
            for phat, y in zip(probs.cpu().detach().numpy(), ytrue.cpu().numpy())])
        term = self.distance_loss(probs, probs_log, closest_unimodal)
        return ce(ypred, ytrue) + self.lamda*term

class WassersteinUnimodal_Wass(WassersteinUnimodal_KLDIV):
    def distance_loss(self, phat, phat_log, target):
        return emd(phat, target)
