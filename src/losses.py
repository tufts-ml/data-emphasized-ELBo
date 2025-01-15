import math
# PyTorch
import torch

def trace_of_Woodbury_matrix_identity(A, U, V):
    # A is n×n, C is k×k, U is n×k, and V is k×n
    # This function assumes A is diagonal and C is a kxk identity matrix.
    n, k = U.shape
    C_inv = torch.eye(k).to(A.device)
    assert A.shape == torch.Size([n]) and V.shape == torch.Size([k, n]), \
    f'AssertionError: Expected A to be of shape torch.Size([{n}]), got {A.shape}. ' \
    f'Expected V to be of shape torch.Size([{k}, {n}]), got {V.shape}.'
    A_inv_U = (1/A).view(-1, 1)*U
    V_A_inv = V*(1/A)
    V_A_inv_U = V_A_inv@U
    # Tr(A+B) == Tr(A)+Tr(B)
    # Tr(AB) == Tr(BA)
    trace = ((1/A).sum() - torch.trace(torch.linalg.inv(C_inv + V_A_inv_U)@V_A_inv@A_inv_U))
    return trace

def log_matrix_determinant_lemma(A, U, V):
    # A is n×n, U is n×k, and V is k×n
    # This function assumes A is diagonal and C is a kxk identity matrix.
    n, k = U.shape
    assert A.shape == torch.Size([n]) and V.shape == torch.Size([k, n]), \
    f'AssertionError: Expected A to be of shape torch.Size([{n}]), got {A.shape}. ' \
    f'Expected V to be of shape torch.Size([{k}, {n}]), got {V.shape}.'
    V_A_inv = V*(1/A)
    V_A_inv_U = V_A_inv@U
    # When A is a diagonal matrix det(A) = \prod a_ii.
    return torch.log(torch.det(torch.eye(k).to(A.device) + V_A_inv_U))+torch.log(A).sum()

def squared_Mahalanobis_distance_of_Woodbury_matrix_identity(x, mu, A, U, V):
    # A is n×n, C is k×k, U is n×k, and V is k×n
    # This function assumes A is diagonal and C is a kxk identity matrix.
    n, k = U.shape
    C_inv = torch.eye(k).to(A.device)
    assert A.shape == torch.Size([n]) and V.shape == torch.Size([k, n]), \
    f'AssertionError: Expected A to be of shape torch.Size([{n}]), got {A.shape}. ' \
    f'Expected V to be of shape torch.Size([{k}, {n}]), got {V.shape}.'
    A_inv_U = (1/A).view(-1, 1)*U
    V_A_inv = V*(1/A)
    V_A_inv_U = V_A_inv@U
    X = A_inv_U
    Y = torch.linalg.inv(C_inv + V_A_inv_U)@V_A_inv
    return ((x-mu)*(1/A))@(x-mu) - ((x-mu)@X)@(Y@(x-mu))

class ERMLoss(torch.nn.Module):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, labels, logits, params, N=1):
        nll = self.criterion(logits, labels)
        return {'loss': nll, 'nll': nll}
    
class L2ZeroLoss(torch.nn.Module):
    def __init__(self, alpha, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, labels, logits, params, N=1):
        nll = self.criterion(logits, labels)
        log_prob = (self.alpha/2) * (params**2).sum()
        return {'nll': nll, 'log_prob': log_prob, 'loss': nll + log_prob}

class L2SPLoss(torch.nn.Module):
    def __init__(self, alpha, bb_loc, beta, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.bb_loc = bb_loc
        self.beta = beta
        self.criterion = criterion
        self.D = len(self.bb_loc)

    def forward(self, labels, logits, params, N=1):
        nll = self.criterion(logits, labels)
        bb_log_prob = (self.alpha/2) * ((params[:self.D] - self.bb_loc.to(params.device))**2).sum()
        clf_log_prob = (self.beta/2) * (params[self.D:]**2).sum()
        return {'bb_log_prob': bb_log_prob, 'clf_log_prob': clf_log_prob, 'nll': nll, 'loss': nll + bb_log_prob + clf_log_prob}
    
class PTYLLoss(torch.nn.Module):
    def __init__(self, bb_loc, beta, lambd, Q, Sigma_diag, criterion=torch.nn.CrossEntropyLoss(), K=5, prior_eps=0.1):
        super().__init__()
        self.bb_loc = bb_loc
        self.beta = beta
        self.criterion = criterion
        self.K = K
        self.lambd = lambd
        self.prior_eps = prior_eps
        self.Q = Q
        self.Sigma_diag = Sigma_diag
        self.D = len(self.bb_loc)
        self.bb_cov_diag = (1/2)*self.Sigma_diag+self.prior_eps
        self.bb_cov_factor = math.sqrt(1/(2*self.K-2))*self.Q[:,:self.K]
        self.bb_prior = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
            loc=self.bb_loc,
            cov_factor=math.sqrt(self.lambd)*self.bb_cov_factor,
            cov_diag=self.lambd*self.bb_cov_diag,
        )
        
    def forward(self, labels, logits, params, N=1):
        nll = self.criterion(logits, labels)
        bb_log_prob = self.bb_prior.log_prob(params[:self.D]).sum()
        bb_log_prob = torch.clamp(bb_log_prob, min=-1e20, max=1e20)
        clf_log_prob = (self.beta/2) * (params[self.D:]**2).sum()
        return {'bb_log_prob': bb_log_prob, 'clf_log_prob': clf_log_prob, 'nll': nll, 'loss': nll - (1/N) * bb_log_prob + clf_log_prob}

class KappaELBoLoss(torch.nn.Module):
    def __init__(self, kappa, sigma_param, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion
        self.kappa = kappa
        self.sigma_param = sigma_param

    def forward(self, labels, logits, params, N=1):
        nll = self.criterion(logits, labels)
        loc_diff_norm = (params**2).sum()
        lambda_star = (loc_diff_norm/len(params)) + torch.nn.functional.softplus(self.sigma_param)**2
        term1 = (torch.nn.functional.softplus(self.sigma_param)**2/lambda_star) * len(params)
        term2 = (1/lambda_star) * loc_diff_norm
        term3 = (len(params) * torch.log(lambda_star)) - (len(params) * torch.log(torch.nn.functional.softplus(self.sigma_param)**2))
        kl = (1/2) * (term1 + term2 - len(params) + term3)
        return {'kl': kl, 'lambda_star': lambda_star, 'loss': nll + (1/self.kappa) * (1/N) * kl, 'nll': nll}    
    
class L2KappaELBoLoss(torch.nn.Module):
    def __init__(self, bb_loc, kappa, sigma_param, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.bb_loc = bb_loc
        self.criterion = criterion
        self.kappa = kappa
        self.sigma_param = sigma_param
        self.D = len(self.bb_loc)
    
    def forward(self, labels, logits, params, N=1):
        nll = self.criterion(logits, labels)
        bb_loc_diff_norm = ((self.bb_loc - params[:self.D])**2).sum()
        clf_loc_diff_norm = (params[self.D:]**2).sum()
        lambda_star = (bb_loc_diff_norm/self.D) + torch.nn.functional.softplus(self.sigma_param)**2
        tau_star = (clf_loc_diff_norm/len(params[self.D:])) + torch.nn.functional.softplus(self.sigma_param)**2
        bb_term1 = (torch.nn.functional.softplus(self.sigma_param)**2/lambda_star) * self.D
        bb_term2 = (1/lambda_star) * bb_loc_diff_norm
        bb_term3 = (self.D * torch.log(lambda_star)) - (self.D * torch.log(torch.nn.functional.softplus(self.sigma_param)**2))
        bb_kl = (1/2) * (bb_term1 + bb_term2 - self.D + bb_term3)
        clf_term1 = (torch.nn.functional.softplus(self.sigma_param)**2/tau_star) * len(params[self.D:])
        clf_term2 = (1/tau_star) * clf_loc_diff_norm
        clf_term3 = (len(params[self.D:]) * torch.log(tau_star)) - (len(params[self.D:]) * torch.log(torch.nn.functional.softplus(self.sigma_param)**2))
        clf_kl = (1/2) * (clf_term1 + clf_term2 - len(params[self.D:]) + clf_term3)
        return {'bb_kl': bb_kl, 'clf_kl': clf_kl, 'lambda_star': lambda_star, 'loss': nll + (1/self.kappa) * (1/N) * (bb_kl + clf_kl), 'nll': nll, 'tau_star': tau_star}
    
class PTYLKappaELBoLoss(torch.nn.Module):
    def __init__(self, bb_loc, kappa, Q, Sigma_diag, sigma_param, criterion=torch.nn.CrossEntropyLoss(), K=5, prior_eps=0.1):
        super().__init__()
        self.bb_loc = bb_loc
        self.criterion = criterion
        self.K = K
        self.kappa = kappa
        self.prior_eps = prior_eps
        self.Q = Q
        self.Sigma_diag = Sigma_diag
        self.sigma_param = sigma_param
        self.D = len(self.bb_loc)
        self.cov_diag = (1/2) * self.Sigma_diag+self.prior_eps
        self.cov_factor = math.sqrt(1/(2*self.K-2)) * self.Q[:,:self.K]
        self.trace_of_the_inverse = trace_of_Woodbury_matrix_identity(self.cov_diag, self.cov_factor, self.cov_factor.T)
        self.log_det = log_matrix_determinant_lemma(self.cov_diag, self.cov_factor, self.cov_factor.T)

    def forward(self, labels, logits, params, N=1):
        nll = self.criterion(logits, labels)
        bb_loc_diff_norm = squared_Mahalanobis_distance_of_Woodbury_matrix_identity(self.bb_loc, params[:self.D], self.cov_diag, self.cov_factor, self.cov_factor.T)
        clf_loc_diff_norm = (params[self.D:]**2).sum()
        lambda_star = (bb_loc_diff_norm + (torch.nn.functional.softplus(self.sigma_param)**2 * self.trace_of_the_inverse)) / self.D
        tau_star = (clf_loc_diff_norm/len(params[self.D:])) + torch.nn.functional.softplus(self.sigma_param)**2
        bb_term1 = (torch.nn.functional.softplus(self.sigma_param)**2/lambda_star) * self.trace_of_the_inverse
        bb_term2 = (1/lambda_star) * bb_loc_diff_norm
        bb_term3 = (self.D * torch.log(lambda_star)) + self.log_det - (self.D * torch.log(torch.nn.functional.softplus(self.sigma_param)**2))
        bb_kl = (1/2) * (bb_term1 + bb_term2 - self.D + bb_term3)
        clf_term1 = (torch.nn.functional.softplus(self.sigma_param)**2/tau_star) * len(params[self.D:])
        clf_term2 = (1/tau_star) * clf_loc_diff_norm
        clf_term3 = (len(params[self.D:]) * torch.log(tau_star)) - (len(params[self.D:]) * torch.log(torch.nn.functional.softplus(self.sigma_param)**2))
        clf_kl = (1/2) * (clf_term1 + clf_term2 - len(params[self.D:]) + clf_term3)
        return {'bb_kl': bb_kl, 'clf_kl': clf_kl, 'lambda_star': lambda_star, 'loss': nll + (1/self.kappa) * (1/N) * (bb_kl + clf_kl), 'nll': nll, 'tau_star': tau_star}
    