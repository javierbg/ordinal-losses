import torch
from torch import nn
import torch.nn.functional as F

#################### UTILITIES ####################

def fact(x):
    return torch.exp(torch.lgamma(x+1))

def log_fact(x):
    return torch.lgamma(x+1)


#################### LOWER-LEVEL ####################

ce = nn.CrossEntropyLoss()

def entropy_loss(Yhat):
    # https://en.wikipedia.org/wiki/Entropy_(information_theory)
    P = F.softmax(Yhat, -1)
    logP = F.log_softmax(Yhat, -1)
    N = P.shape[0]
    return -torch.sum(P * logP) / N

def neighbor_loss(margin, Yhat, Y):
    margin = torch.tensor(margin, device=device)
    P = F.softmax(Yhat, -1)
    K = P.shape[1]
    loss = 0
    for k in range(K-1):
        # force previous probability to be superior to next
        reg_gt = (Y >= k+1).float() * F.relu(margin+P[:,  k]-P[:, k+1])
        reg_lt = (Y <= k).float() * F.relu(margin+P[:, k+1]-P[:, k])
        loss += torch.mean(reg_gt + reg_lt)
    return loss

#################### HIGHER-LEVEL ####################

class CrossEntropy:
    def __init__(self, K):
        self.K = K

    def how_many_outputs(self):
        # how many output neurons does this loss require?
        return self.K

    def __call__(self, Yhat, Y):
        # computes the loss
        return ce(Yhat, Y)

    def to_proba(self, Yhat):
        # call output -> probabilities
        return F.softmax(Yhat, 1)

    def to_classes(self, Phat, type):
        # probabilities -> classes
        assert type in ('mode', 'mean', 'median')
        if type == 'mode':
            return Phat.argmax(1)
        if type == 'mean':  # so-called expectation trick
            kk = torch.arange(args.classes, device=Phat.device)
            return torch.round(torch.sum(Yhat * kk, 1)).long()
        if type == 'median':
            # the weighted median is the value whose cumulative probability is 0.5
            Pc = torch.cumsum(Phat, 1)
            return torch.sum(Pc < 0.5, 1)

class OrdinalEncoding(CrossEntropy):
    # Reference: https://arxiv.org/pdf/0704.1028.pdf
    def how_many_outputs(self):
        return self.K-1

    def __call__(self, Yhat, Y):
        # if K=4, then
        #     Y=0 => Y_=[0, 0, 0]
        #     Y=1 => Y_=[1, 0, 0]
        #     Y=2 => Y_=[1, 1, 0]
        #     Y=3 => Y_=[1, 1, 1]
        KK = torch.arange(self.K-1, device=Y.device).expand(Y.shape[0], -1)
        YY = (Y[:, None] > KK).float()
        return F.binary_cross_entropy_with_logits(Yhat, YY)

    def to_proba(self, Yhat):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y >= k) into P(Y = k)
        # P(Y=0) = 1-P(Y≥1)
        # P(Y=1) = P(Y≥1)-P(Y≥2)
        # ...
        # P(Y=K-1) = P(Y≥K-1)
        Phat = torch.sigmoid(Yhat)
        Phat = torch.cat((1-Phat[:, :1], Phat[:, :-1] - Phat[:, 1:], Phat[:, -1:]), 1)
        return torch.clamp(Phat, 0, 1)

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
    def activation(self, x):
        # they apply softplus (relu) to avoid log(negative)
        x = nn.Softplus()(x)
        KK = torch.arange(1., self.K+1, device=x.device)
        return KK*torch.log(x) - x - log_fact(KK)

    def __call__(self, Yhat, Y):
        Yhat = self.activation(Yhat)
        return ce(Yhat, Y)

    def to_proba(self, Yhat):
        Yhat = self.activation(Yhat)
        return super.to_proba(Yhat)

# Our losses.
# Notice that the following constructors require extra parameters.
# Reference: https://peerj.com/articles/cs-457/

class OurLosses(CrossEntropy):
    def __init__(self, K, lamda, omega):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

class CO2(OurLosses):
    def __call__(self, Yhat, Y):
        return self.lamda*ce(Yhat, Y) + neighbor_loss(self.omega, Yhat, Y)

class CO(CO2):
    def __init__(self, pretrained_model, K, lambda_, omega):
        # CO is CO2 with omega=0
        # for convenience, we still receive an omega, but we ignore it
        super().__init__(pretrained_model, K, lambda_, 0)

class HO2(OurLosses):
    def __call__(self, Yhat, Y):
        return self.lamda*entropy_loss(Yhat) + neighbor_loss(self.omega, Yhat, Y)
