import math
# PyTorch
import torch

def trace_of_Woodbury_matrix_identity(A, U, V):
    # A is n×n, C is k×k, U is n×k, and V is k×n
    # This function assumes A is diagonal and C is a kxk identity matrix.
    n, k = U.shape
    assert A.shape == (n,)
    assert V.shape == (k, n)
    C_inv = torch.eye(k, device=A.device)
    A_inv_U = (1 / A).view(-1, 1) * U
    V_A_inv = V * (1 / A)
    V_A_inv_U = V_A_inv @ U
    # Tr(A+B) == Tr(A)+Tr(B)
    # Tr(AB) == Tr(BA)
    trace = ((1 / A).sum() - torch.trace(torch.linalg.inv(C_inv + V_A_inv_U) @ V_A_inv @ A_inv_U))
    return trace

def diag_of_Woodbury_matrix_identity(A, U, V):
    # A is n×n, C is k×k, U is n×k, and V is k×n
    # This function assumes A is diagonal and C is a kxk identity matrix.
    n, k = U.shape
    assert A.shape == (n,)
    assert V.shape == (k, n)
    C_inv = torch.eye(k, device=A.device)
    A_inv_U = (1 / A).view(-1, 1) * U
    V_A_inv = V * (1 / A)
    V_A_inv_U = V_A_inv @ U
    diag = (1 / A) - (A_inv_U * (torch.linalg.inv(C_inv + V_A_inv_U) @ V_A_inv).T).sum(dim=1) 
    return diag

def log_matrix_determinant_lemma(A, U, V):
    # A is n×n, U is n×k, and V is k×n
    # This function assumes A is diagonal and C is a kxk identity matrix.
    n, k = U.shape
    assert A.shape == (n,)
    assert V.shape == (k, n)
    I_k = torch.eye(k, device=A.device)
    V_A_inv = V * (1 / A)
    V_A_inv_U = V_A_inv @ U
    # When A is a diagonal matrix det(A) = \prod a_ii.
    return torch.log(torch.det(I_k + V_A_inv_U))+torch.log(A).sum()

def squared_Mahalanobis_distance_of_Woodbury_matrix_identity(x, A, U, V):
    # A is n×n, C is k×k, U is n×k, and V is k×n
    # This function assumes A is diagonal and C is a kxk identity matrix.
    n, k = U.shape
    assert A.shape == (n,)
    assert V.shape == (k, n)
    C_inv = torch.eye(k, device=A.device)
    A_inv_U = (1 / A).view(-1, 1) * U
    V_A_inv = V * (1 / A)
    V_A_inv_U = V_A_inv @ U
    X = A_inv_U
    Y = torch.linalg.inv(C_inv + V_A_inv_U) @ V_A_inv
    return (x * (1 / A)) @ x - (x @ X) @ (Y @ x)

class NoPrior(torch.nn.Module):
    def __init__(self, num_params):
        super().__init__()
        self.num_params = num_params
        
    def kl(self, params, sigma):
        return torch.tensor(0.0, dtype=params.dtype, device=params.device)

    def log_prob(self, params):
        return torch.tensor(0.0, dtype=params.dtype, device=params.device)

class IsotropicGaussianPrior(torch.nn.Module):
    def __init__(self, num_params=None, prior_params=None, prior_variance=1.0):
        super().__init__()
        self.prior_params = prior_params
        self.num_params = len(self.prior_params) if num_params is None else num_params
        self.prior_variance = torch.tensor(prior_variance, dtype=torch.float32)

    def kl(self, params, sigma):
        assert len(params) == self.num_params
        params_diff_norm = (params**2).sum() if self.prior_params is None else ((params - self.prior_params)**2).sum()
        if sigma.shape == ():
            prior_variance = (params_diff_norm + sigma**2 * self.num_params) / self.num_params
            trace = (sigma**2 / prior_variance) * self.num_params
            quad_term = (1 / prior_variance) * params_diff_norm
            log_det = self.num_params * torch.log(prior_variance) - self.num_params * torch.log(sigma**2)
        elif sigma.shape == (self.num_params,):
            prior_variance = (params_diff_norm + (sigma**2).sum()) / self.num_params
            trace = (sigma**2).sum() / prior_variance
            quad_term = (1 / prior_variance) * params_diff_norm
            log_det = self.num_params * torch.log(prior_variance) - torch.log(sigma**2).sum()
        kl = 0.5 * (trace + quad_term - self.num_params + log_det)
        return kl
    
    def log_prob(self, params):
        assert len(params) == self.num_params
        params_diff_norm = (params**2).sum() if self.prior_params is None else ((params - self.prior_params)**2).sum()
        log_norm_const = self.num_params * math.log(2.0 * math.pi)
        log_det = self.num_params * torch.log(self.prior_variance)
        quad_term = (1 / self.prior_variance) * params_diff_norm
        log_prob = -0.5 * (log_norm_const + log_det + quad_term)
        return log_prob
            
class LowRankGaussianPrior(torch.nn.Module):
    def __init__(self, prior_params, Sigma_diag, Q, K=5, prior_eps=0.1, prior_variance=1.0):
        super().__init__()
        self.prior_params = prior_params
        self.num_params = len(self.prior_params)
        self.Sigma_diag = Sigma_diag
        self.Q = Q
        self.K = K
        self.prior_eps = prior_eps
        self.prior_variance = torch.tensor(prior_variance, dtype=torch.float32)
        self.cov_diag = 0.5 * self.Sigma_diag + self.prior_eps
        self.cov_factor = math.sqrt(1 / (2 * self.K - 2)) * self.Q[:,:self.K]
        self.trace_of_cov_inv = trace_of_Woodbury_matrix_identity(self.cov_diag, self.cov_factor, self.cov_factor.T)
        self.diag_of_cov_inv = diag_of_Woodbury_matrix_identity(self.cov_diag, self.cov_factor, self.cov_factor.T)
        self.log_det_cov = log_matrix_determinant_lemma(self.cov_diag, self.cov_factor, self.cov_factor.T)
        
    def kl(self, params, sigma):
        assert len(params) == self.num_params
        params_diff = params if self.prior_params is None else params - self.prior_params
        params_diff_norm = squared_Mahalanobis_distance_of_Woodbury_matrix_identity(params_diff, self.cov_diag, self.cov_factor, self.cov_factor.T)
        if sigma.shape == ():
            prior_variance = (params_diff_norm + sigma**2 * self.trace_of_Sigma_p_inv) / self.num_params
            trace = (sigma**2 / prior_variance) * self.trace_of_the_cov_inv
            quad_term = (1 / prior_variance) * params_diff_norm
            log_det = self.num_params * torch.log(prior_variance) + self.log_det_cov - self.num_params * torch.log(sigma**2)
        elif sigma.shape == (self.num_params,):
            prior_variance = (params_diff_norm + (self.diag_of_cov_inv * sigma**2).sum()) / self.num_params
            trace = (1 / prior_variance) * (self.diag_of_cov_inv * sigma**2).sum()
            quad_term = (1 / prior_variance) * params_diff_norm
            log_det = self.num_params * torch.log(prior_variance) + self.log_det_cov - torch.log((sigma**2).sum())
        kl = 0.5 * (trace + quad_term - self.num_params + log_det)
        return kl
            
    def log_prob(self, params):
        assert len(params) == self.num_params
        params_diff = params if self.prior_params is None else params - self.prior_params
        log_norm_const = self.num_params * math.log(2.0 * math.pi)
        log_det = self.num_params * torch.log(self.prior_variance) + self.log_det_cov
        quad_term = (1 / self.prior_variance) * squared_Mahalanobis_distance_of_Woodbury_matrix_identity(params_diff, self.cov_diag, self.cov_factor, self.cov_factor.T)
        log_prob = -0.5 * (log_norm_const + log_det + quad_term)
        return log_prob
    