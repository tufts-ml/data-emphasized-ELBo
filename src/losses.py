# PyTorch
import torch
import torchmetrics

class MAPLoss(torch.nn.Module):
    def __init__(self, likelihood, prior):
        super().__init__()
        self.likelihood = likelihood
        self.prior = prior

    def forward(self, logits, labels, params, N):
        nll = self.likelihood(logits, labels)
        log_prior = self.prior.log_prob(params)
        return {"log_prior": log_prior, "loss": nll - (1 / N) * log_prior, "nll": nll}
        
class TemperedELBOLoss(torch.nn.Module):
    def __init__(self, model, likelihood, prior, kappa=1.0):
        super().__init__()
        self.kappa = kappa
        self.likelihood = likelihood
        self.model = model
        self.prior = prior

    def forward(self, logits, labels, params, N):
        nll = self.likelihood(logits, labels)
        sigma = torch.nn.functional.softplus(self.model.raw_sigma)
        kl = self.prior.kl(params, sigma)
        return {"kl": kl, "loss": nll + (1 / self.kappa) * (1 / N) * kl, "nll": nll}
    
class TransferLearningLoss(torch.nn.Module):
    def __init__(self, likelihood, backbone_prior, classifier_prior, num_backbone_params):
        super().__init__()
        self.likelihood = likelihood
        self.backbone_prior = backbone_prior
        self.classifier_prior = classifier_prior
        self.num_backbone_params = num_backbone_params

    def forward(self, logits, labels, params, N):
        nll = self.likelihood(logits, labels)
        log_prior = self.backbone_prior.log_prob(params[:self.num_backbone_params]) + self.classifier_prior.log_prob(params[self.num_backbone_params:])
        return {"log_prior": log_prior, "loss": nll - log_prior, "nll": nll}
    
class TransferLearningMAPLoss(torch.nn.Module):
    def __init__(self, likelihood, backbone_prior, classifier_prior, num_backbone_params):
        super().__init__()
        self.likelihood = likelihood
        self.backbone_prior = backbone_prior
        self.classifier_prior = classifier_prior
        self.num_backbone_params = num_backbone_params

    def forward(self, logits, labels, params, N):
        nll = self.likelihood(logits, labels)
        log_prior = self.backbone_prior.log_prob(params[:self.num_backbone_params]) + self.classifier_prior.log_prob(params[self.num_backbone_params:])
        return {"log_prior": log_prior, "loss": nll - (1 / N) * log_prior, "nll": nll}
    
class TransferLearningTemperedELBOLoss(torch.nn.Module):
    def __init__(self, model, likelihood, backbone_prior, classifier_prior, kappa=1.0):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.backbone_prior = backbone_prior
        self.classifier_prior = classifier_prior
        self.kappa = kappa
        
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.likelihood.num_classes, average="macro")
        self.ece = torchmetrics.CalibrationError(task="multiclass", num_classes=self.likelihood.num_classes, n_bins=15, norm="l1")

    def forward(self, logits, labels, params, N):
        nll = self.likelihood(logits, labels)
        sigma = torch.nn.functional.softplus(self.model.raw_sigma)
        if sigma.shape == ():
            kl = self.backbone_prior.kl(params[:self.backbone_prior.num_params], sigma) + self.classifier_prior.kl(params[self.backbone_prior.num_params:], sigma)
        elif sigma.shape == (len(params),):
            kl = self.backbone_prior.kl(params[:self.backbone_prior.num_params], sigma[:self.backbone_prior.num_params]) + self.classifier_prior.kl(params[self.backbone_prior.num_params:], sigma[self.backbone_prior.num_params:])
        with torch.no_grad():
            device = logits.device
            self.acc = self.acc.to(device)
            self.ece = self.ece.to(device)
            acc = self.acc(logits, labels)
            ece = self.ece(logits, labels)
        return {"acc": acc, "ece": ece, "kl": kl, "loss": nll + (1 / self.kappa) * (1 / N) * kl, "nll": nll}
