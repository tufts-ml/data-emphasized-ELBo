import math
# PyTorch
import torch

def trace_of_Woodbury_matrix_identity(A, U, V):
    # A is n×n, C is k×k, U is n×k, and V is k×n
    # This function assumes A is diagonal and C is a kxk identity matrix.
    n, k = U.shape
    C_inv = torch.eye(k).to(A.device)
    n, k = U.shape
    assert A.shape == (n,)
    assert V.shape == (k, n)
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
    C_inv = torch.eye(k).to(A.device)
    assert A.shape == (n,)
    assert V.shape == (k, n)
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
    V_A_inv = V * (1 / A)
    V_A_inv_U = V_A_inv @ U
    # When A is a diagonal matrix det(A) = \prod a_ii.
    return torch.log(torch.det(torch.eye(k).to(A.device) + V_A_inv_U))+torch.log(A).sum()

def squared_Mahalanobis_distance_of_Woodbury_matrix_identity(x, A, U, V):
    # A is n×n, C is k×k, U is n×k, and V is k×n
    # This function assumes A is diagonal and C is a kxk identity matrix.
    n, k = U.shape
    assert A.shape == (n,)
    assert V.shape == (k, n)
    A_inv_U = (1 / A).view(-1, 1) * U
    V_A_inv = V * (1 / A)
    V_A_inv_U = V_A_inv @ U
    X = A_inv_U
    Y = torch.linalg.inv(C_inv + V_A_inv_U) @ V_A_inv
    return (x * (1 / A)) @ x - (x @ X) @ (Y @ x)

class IsotropicGaussianPrior(torch.nn.Module):
    def __init__(self, learnable_tau=False, num_params=None, prior_params=None, tau=1.0, use_tau_star=False):
        super().__init__()
        
        self.num_params = num_params
        self.prior_params = prior_params
        
        if self.num_params is None:
            assert self.prior_params is not None
            self.num_params = len(self.prior_params)
                
        self.learnable_tau = learnable_tau
        self.use_tau_star = use_tau_star
        
        if self.learnable_tau:
            self.raw_tau = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(tau, dtype=torch.float32))))
        else:
            self.register_buffer("raw_tau", torch.log(torch.expm1(torch.tensor(tau, dtype=torch.float32))))
        
    def kl(self, params, sigma):
        
        assert len(params) == self.num_params
                
        if self.prior_params is None:
            params_diff_norm = (params**2).sum()
        else:
            params_diff_norm = ((params - self.prior_params)**2).sum()
        
        if sigma.shape == ():
            if self.use_tau_star:
                tau = (params_diff_norm + sigma**2 * self.num_params) / self.num_params
            else:
                tau = self.tau
            trace = (sigma**2 / tau) * self.num_params
            quad_term = (1 / tau) * params_diff_norm
            log_det = self.num_params * torch.log(tau) - self.num_params * torch.log(sigma**2)
        elif sigma.shape == (self.num_params,):
            if self.use_tau_star:
                tau = (params_diff_norm + (sigma**2).sum()) / self.num_params
            else:
                tau = self.tau
            trace = (sigma**2).sum() / tau
            quad_term = (1 / tau) * params_diff_norm
            log_det = self.num_params * torch.log(tau) - torch.log(sigma**2).sum()

        kl = 0.5 * (trace + quad_term - self.num_params + log_det)
            
        return kl
    
    def log_prob(self, params):
        
        assert len(params) == self.num_params
        
        if self.prior_params is None:
            params_diff_norm = (params**2).sum()
        else:
            params_diff_norm = ((params - self.prior_params)**2).sum()
            
        log_norm_const = -0.5 * self.num_params * math.log(2.0 * math.pi)
        log_det = -0.5 * self.num_params * torch.log(self.tau)
        quad_term = -0.5 * params_diff_norm / self.tau
        log_prob = log_norm_const + log_det + quad_term
            
        return log_prob
    
    @property
    def tau(self):
        return torch.nn.functional.softplus(self.raw_tau)
        
class LowRankGaussianPrior(torch.nn.Module):
    def __init__(self, prior_params, Sigma_diag, Q, K=5, learnable_tau=False, prior_eps=0.1, tau=1.0):
        super().__init__()
        
        self.prior_params = prior_params
        self.num_params = len(self.prior_params)
        self.Sigma_diag = Sigma_diag
        self.Q = Q
        self.K = K
        self.prior_eps = prior_eps
                        
        self.learnable_tau = learnable_tau
        
        if self.learnable_tau:
            self.raw_tau = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(tau, dtype=torch.float32))))
        else:
            self.register_buffer("raw_tau", torch.log(torch.expm1(torch.tensor(tau, dtype=torch.float32))))

        self.cov_diag = 0.5 * self.Sigma_diag + self.prior_eps
        self.cov_factor = math.sqrt(1 / (2 * self.K - 2)) * self.Q[:,:self.K]
        self.trace_of_cov_inv = trace_of_Woodbury_matrix_identity(self.cov_diag, self.cov_factor, self.cov_factor.T)
        self.diag_of_cov_inv = diag_of_Woodbury_matrix_identity(self.cov_diag, self.cov_factor, self.cov_factor.T)
        self.log_det_cov = log_matrix_determinant_lemma(self.cov_diag, self.cov_factor, self.cov_factor.T)
        
    def kl(self, params, sigma):

        assert len(params) == self.num_params
        
        if self.prior_params is None:
            params_diff = params
        else:
            params_diff = params - self.prior_params
            
        params_diff_norm = squared_Mahalanobis_distance_of_Woodbury_matrix_identity(params_diff, self.cov_diag, self.cov_factor, self.cov_factor.T)
        
        if sigma.shape == ():
            #tau_star = (params_diff_norm + sigma**2 * self.trace_of_Sigma_p_inv) / self.num_params
            trace = (sigma**2 / self.tau) * self.trace_of_the_cov_inv
            quad_term = (1 / self.tau) * params_diff_norm
            log_det = self.num_params * torch.log(self.tau) + self.log_det_cov - self.num_params * torch.log(sigma**2)
        elif sigma.shape == (self.num_params,):
            #tau_star = (params_diff_norm + (self.diag_of_cov_inv * sigma**2).sum()) / self.num_params
            trace = (1 / self.tau) * (self.diag_of_cov_inv * sigma**2).sum()
            quad_term = (1 / self.tau) * params_diff_norm
            log_det = self.num_params * torch.log(self.tau) + self.log_det_cov - torch.log((sigma**2).sum())
            
        kl = 0.5 * (trace + quad_term - self.num_params + log_det)
        
        return kl
            
    def log_prob(self, params):
        
        assert len(params) == self.num_params
        
        if self.prior_params is None:
            params_diff = params
        else:
            params_diff = params - self.prior_params
            
        log_norm_const = -0.5 * self.num_params * math.log(2.0 * math.pi)
        log_det = -0.5 * (self.num_params * torch.log(self.tau) + self.log_det_cov)
        quad_term = -0.5 * squared_Mahalanobis_distance_of_Woodbury_matrix_identity(params_diff, self.cov_diag, self.cov_factor, self.cov_factor.T) / self.tau
        log_prob = log_norm_const + log_det + quad_term
        
        return log_prob
    
    @property
    def lambd(self):
        return torch.nn.functional.softplus(self.raw_tau)
        