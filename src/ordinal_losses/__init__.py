import torch
from torch import nn
import torch.nn.functional as F

#################### UTILITIES ####################

def fact(x):
    return torch.exp(torch.lgamma(x+1))

def log_fact(x):
    return torch.lgamma(x+1)

# we are using softplus instead of relu (like http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf)
# since it is smoother to optimize.
approx_relu = F.softplus

#################### LOWER-LEVEL ####################

ce = nn.CrossEntropyLoss()

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

#################### HIGHER-LEVEL ####################

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

    def to_classes(self, Phat, method='mode'):
        # probabilities -> classes
        assert method in ('mode', 'mean', 'median')
        if method == 'mode':
            return Phat.argmax(1)
        if method == 'mean':  # so-called expectation trick
            kk = torch.arange(args.classes, device=Phat.device)
            return torch.round(torch.sum(Yhat * kk, 1)).long()
        if method == 'median':
            # the weighted median is the value whose cumulative probability is 0.5
            Pc = torch.cumsum(Phat, 1)
            return torch.sum(Pc < 0.5, 1)

class OrdinalEncoding(CrossEntropy):
    # Reference: https://arxiv.org/pdf/0704.1028.pdf
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
        return ce(Yhat, YY)

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

class BinomialUnimodal_CE(CrossEntropy):
    # Reference: https://www.sciencedirect.com/science/article/pii/S089360800700202X
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

class PoissonUnimodal(CrossEntropy):
    # Reference: http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf
    def how_many_outputs(self):
        return 1

    def activation(self, Yhat):
        # they apply softplus (relu) to avoid log(negative)
        Yhat = F.softplus(Yhat)
        KK = torch.arange(1., self.K+1, device=Yhat.device)[None]
        return KK*torch.log(Yhat) - Yhat - log_fact(KK)

class CDW_CE(CrossEntropy):
    # Reference: https://arxiv.org/pdf/2202.05167.pdf
    def __init__(self, K, alpha=5):
        super().__init__(K)
        self.alpha = alpha

    def __call__(self, Yhat, Y):
        log_1minus_Yhat = F.log_softmax(-Yhat, 1)
        return -torch.mean(log_1minus_Yhat * torch.abs(torch.arange(self.K, device=Y.device)[None]-Y[:, None])**self.alpha)

# Our losses that promote unimodality.
# Notice that our losses require extra parameters: lamda and omega.
# Reference: https://peerj.com/articles/cs-457/

class OurLosses(CrossEntropy):
    def __init__(self, K, lamda=0.1, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

class CO2(OurLosses):
    # CO is the same with omega=0
    def __call__(self, Yhat, Y):
        return ce(Yhat, Y) + self.lamda*neighbor_term(Yhat, Y, self.omega)

class CO(CO2):
    # CO is the same as CO2 with omega=0
    def __init__(self, K, lamda=0.1, omega=0.05):
        super().__init__(K, lamda, 0)

class HO2(OurLosses):
    def __call__(self, Yhat, Y):
        return entropy_term(Yhat) + self.lamda*neighbor_term(Yhat, Y, self.omega)

# Our losses that promote quasi-unimodality (they are more forgiving than the
# previous ones).
# Reference: https://www.mdpi.com/2227-7390/10/6/980

class QUL(OurLosses):
    def __call__(self, Yhat, Y):
        return quasi_neighbor_term(Yhat, Y, self.omega)

class QUL_CE(OurLosses):
    def __call__(self, Yhat, Y):
        return ce(Yhat, Y) + self.lamda*quasi_neighbor_term(Yhat, Y, self.omega)

class QUL_HO(OurLosses):
    def __call__(self, Yhat, Y):
        return entropy_term(Yhat) + self.lamda*quasi_neighbor_term(Yhat, Y, self.omega)
