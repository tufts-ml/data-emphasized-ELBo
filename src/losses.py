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
        loss = nll - (1 / N) * log_prior
        return {"log_prior": log_prior, "loss": loss, "nll": nll}
        
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
        loss = nll + (1 / self.kappa) * (1 / N) * kl
        return {"kl": kl, "loss": loss, "nll": nll}
        
class TransferLearningLoss(torch.nn.Module):
    def __init__(self, model, likelihood, backbone_prior, classifier_prior):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.backbone_prior = backbone_prior
        self.classifier_prior = classifier_prior

    def forward(self, logits, labels, params, N):
        nll = self.likelihood(logits, labels)
        backbone_params = params[:self.backbone_prior.num_params]
        classifier_params = params[self.backbone_prior.num_params:]
        log_prob = self.backbone_prior.log_prob(backbone_params) + self.classifier_prior.log_prob(classifier_params)
        loss = nll - log_prob
        return {"log_prob": log_prob, "loss": loss, "nll": nll}
    
class TransferLearningMAPLoss(torch.nn.Module):
    def __init__(self, model, likelihood, backbone_prior, classifier_prior):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.backbone_prior = backbone_prior
        self.classifier_prior = classifier_prior

    def forward(self, logits, labels, params, N):
        nll = self.likelihood(logits, labels)
        backbone_params = params[:self.backbone_prior.num_params]
        classifier_params = params[self.backbone_prior.num_params:]
        log_prob = self.backbone_prior.log_prob(backbone_params) + self.classifier_prior.log_prob(classifier_params)
        loss = nll - (1 / N) * log_prob
        return {"log_prob": log_prob, "loss": loss, "nll": nll}
        
class TransferLearningTemperedELBOLoss(torch.nn.Module):
    def __init__(self, model, likelihood, backbone_prior, classifier_prior, kappa=1.0):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.backbone_prior = backbone_prior
        self.classifier_prior = classifier_prior
        self.kappa = kappa

    def forward(self, logits, labels, params, N):
        nll = self.likelihood(logits, labels)
        backbone_params = params[:self.backbone_prior.num_params]
        classifier_params = params[self.backbone_prior.num_params:]
        sigma = torch.nn.functional.softplus(self.model.raw_sigma)
        if sigma.shape == ():
            kl = self.backbone_prior.kl(backbone_params, sigma) + self.classifier_prior.kl(classifier_params, sigma)
        elif sigma.shape == (len(params),):
            backbone_sigma = sigma[:self.backbone_prior.num_params]
            classifier_sigma = sigma[self.backbone_prior.num_params:]
            kl = self.backbone_prior.kl(backbone_params, backbone_sigma) + self.classifier_prior.kl(classifier_params, classifier_sigma)
        loss = nll + (1 / self.kappa) * (1 / N) * kl
        return {"kl": kl, "loss": loss, "nll": nll}
