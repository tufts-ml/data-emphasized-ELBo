# PyTorch
import torch

class VariationalLinear(torch.nn.Module):
    def __init__(self, layer, sigma_param, use_posterior=False):
        super().__init__()
        self.layer = layer
        self.sigma_param = sigma_param
        self.use_posterior = use_posterior
        
    def forward(self, x):
        if self.training or self.use_posterior:
            return torch.nn.functional.linear(
                x,
                self.variational_weight,
                self.variational_bias,
            )
            
        return self.layer(x)
    
    @property
    def variational_weight(self):
        return self.layer.weight + torch.nn.functional.softplus(self.sigma_param) * torch.randn_like(self.layer.weight).to(self.layer.weight.device)
            
    @property
    def variational_bias(self):
        return self.layer.bias + torch.nn.functional.softplus(self.sigma_param) * torch.randn_like(self.layer.bias).to(self.layer.bias.device) if self.layer.bias is not None else None

class VariationalConv2d(VariationalLinear):
    def __init__(self, layer, sigma_param, use_posterior=False):
        super().__init__(layer, sigma_param, use_posterior)
        
    def forward(self, x):
        if self.training or self.use_posterior:
            return torch.nn.functional.conv2d(
                x,
                self.variational_weight,
                self.variational_bias,
                self.layer.stride,
                self.layer.padding,
                self.layer.dilation,
                self.layer.groups
            )
        
        return self.layer(x)

class VariationalBatchNorm2d(VariationalLinear):
    def __init__(self, layer, sigma_param, use_posterior=False):
        super().__init__(layer, sigma_param, use_posterior)
        
    def forward(self, x):
        if self.training or self.use_posterior:

            if self.layer.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.layer.momentum

            if self.layer.training and self.layer.track_running_stats:
                if self.layer.num_batches_tracked is not None:
                    self.layer.num_batches_tracked.add_(1)
                    if self.layer.momentum is None:
                        exponential_average_factor = 1.0 / float(self.layer.num_batches_tracked)
                    else:
                        exponential_average_factor = self.layer.momentum

            if self.layer.training:
                bn_training = True
            else:
                bn_training = (self.layer.running_mean is None) and (self.layer.running_var is None)

            return torch.nn.functional.batch_norm(
                x, 
                self.layer.running_mean if not self.layer.training or self.layer.track_running_stats else None, 
                self.layer.running_var if not self.layer.training or self.layer.track_running_stats else None, 
                self.variational_weight,
                self.variational_bias,
                bn_training, 
                exponential_average_factor, 
                self.layer.eps, 
            )
        
        return self.layer(x)
    