# PyTorch
import torch

class GaussianLikelihood(torch.nn.Module):
    def __init__(self, num_classes, learnable_noise=False, noise=1.0):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.learnable_noise = learnable_noise
        
        if self.learnable_noise:
            self.raw_noise = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(noise, dtype=torch.float32))))
        else:
            self.register_buffer("raw_noise", torch.log(torch.expm1(torch.tensor(noise, dtype=torch.float32))))
    
    def forward(self, logits, labels, reduction="mean"):
        device = logits.device
        batch_size = len(logits)
        var = self.noise**2 * torch.ones(size=(batch_size,), device=device)
        return torch.nn.functional.gaussian_nll_loss(logits, labels, var, reduction=reduction, full=True)

    @property
    def noise(self):
        return torch.nn.functional.softplus(self.raw_noise)
    
class CategoricalLikelihood(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.num_classes = num_classes
            
    def forward(self, logits, labels, reduction="mean"):
        return torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)
