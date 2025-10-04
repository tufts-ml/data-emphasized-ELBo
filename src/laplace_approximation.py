import math
import copy
# PyTorch
import torch
# Importing our custom module(s)
import utils


class DiagEFLaplace(torch.nn.Module):
    def __init__(self, model, likelihood, prior, temp=1.0):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.prior = prior
        self.temp = temp
        self.nll = 0.0
        self.H = None
    
    def _compute_H(self, dataloader):
        assert dataloader.batch_size == 1
        self.nll = 0.0
        self.H = torch.zeros_like(utils.flatten_params(self.model))
        for images_i, labels_i in dataloader:
            self.model.zero_grad()
            logits_i = self.model(images_i)
            nll_i = self.likelihood(logits_i, labels_i)
            nll_i.backward()
            self.nll += nll_i.item()
            grads_i = copy.deepcopy(utils.flatten_grads(self.model).detach())
            H_i = grads_i**2
            self.H += H_i

    def lml_loss(self, logits, labels, params, N):
        assert len(logits) == N
        nll = self.likelihood(logits, labels, reduction="sum")
        D = len(params)
        log_prior = self.prior.log_prob(params)
        norm_const = 0.5 * D * math.log(2 * math.pi)
        log_det = 0.5 * torch.log(self.H + (1 / self.prior.prior_variance)).sum()
        lml = -nll + log_prior + norm_const - log_det
        loss = (1 / N) * -lml
        return {"loss": loss}
    
    def sample(self):
        params = utils.flatten_params(self.model)
        eps = torch.randn_like(params)
        posterior_var = 1 / (self.H + (1 / self.prior.prior_variance))
        sample = params + self.temp * torch.sqrt(posterior_var) * eps
        return sample
    
    def sample_forward(self, X):
        with torch.no_grad():
            sampled_params = self.sample()
            params = copy.deepcopy(utils.flatten_params(self.model))
            utils.unflatten_params(self.model, sampled_params)
            logits = self.model(X)
            utils.unflatten_params(self.model, params)
            return logits
        