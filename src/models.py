# PyTorch
import torch
# Laplace
import laplace

class L2SPLaplace(laplace.baselaplace.DiagLaplace):

    @property
    def prior_precision(self) -> torch.Tensor:
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision: float | torch.Tensor):
        self._posterior_scale = None
        self._prior_precision = prior_precision.to(
            device=self._device, dtype=self._dtype
        )
        
    @property
    def prior_precision_diag(self) -> torch.Tensor:
        diag = torch.empty(self.n_params, device=self._device, dtype=self._dtype)
        diag[:self.num_backbone_params].fill_(self.prior_precision[0])
        diag[self.num_backbone_params:].fill_(self.prior_precision[1])
        return diag
    