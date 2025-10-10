# data-emphasized-ELBO
[Learning Hyperparameters via a Data-Emphasized Variational Objective](https://arxiv.org/abs/2502.01861) by Ethan Harvey, Mikhail Petrov, and Michael C. Hughes

![Figure 2](./notebooks/computational_time_comparison.png)
Figure 3: Test accuracy over time for L2-SP transfer learning methods. We run each method on 3 separate train sets of size $N$ (3 marker styles). Each panel shows a distinct task: ConvNeXt-Tiny fine-tuned on CIFAR-10, Flower-102, and Pet-37 and BERT-base fine-tuned on News-4. We compare MAP + GS, MAP + BO, diagEF LA-LML (Immer et al., 2021), diagEF LA-CLML (Lotfi et al., 2022), iso ELBO, and iso DE-ELBO (ours). **Takeaway: After just a few hours, iso DE-ELBO reaches as good or better performance at small data sizes and similar performance at large sizes, even when other methods are given many additional hours.**

## Installing environment
```
conda env create -f l3d_24f_cuda.yml
conda activate l3d_2024f_cuda12_1
```

## L2-SP example
```python
# Load pretrained model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Identity()
backbone_prior_params = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach()).to(device)
model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
num_classifier_params = (2048 * num_classes) + num_classes

# Define priors
backbone_prior = priors.IsotropicGaussianPrior(prior_params=backbone_prior_params, prior_variance=1.0)
classifier_prior = priors.IsotropicGaussianPrior(num_params=num_classifier_params, prior_variance=1.0)

# Define likelihood
likelihood = likelihoods.CategoricalLikelihood()

# Define approximate posterior variance
model.raw_sigma = torch.nn.Parameter(utils.inv_softplus(torch.tensor(1e-4, device=device)))
utils.add_variational_layers(model, model.raw_sigma)
model.use_posterior = types.MethodType(utils.use_posterior, model)

# Define loss
criterion = losses.TransferLearningTemperedELBOLoss(model, likelihood, backbone_prior, classifier_prior, kappa=D/N)

# Train
for epoch in range(epochs):
    ...
```

## Reproducibility
The notebook `regression_demo.ipynb` can be used to reproduce random Fourier feature experiments (Fig. 2). The scripts can be used to reproduce transfer learning experiments. To reproduce transfer learning figures without retraining models, download experiments and run notebooks.

## Citation
```bibtex
@article{harvey2025learning,
  author={Harvey, Ethan and Petrov, Mikhail and Hughes, Michael C},
  title={Learning Hyperparameters via a Data-Emphasized Variational Objective},
  journal={arXiv preprint arXiv:2502.01861},
  year={2025},
}
```
```bibtex
@inproceedings{harvey2024learning,
  author={Ethan Harvey and Mikhail Petrov and Michael C Hughes},
  title={Learning the Regularization Strength for Deep Fine-Tuning via a Data-Emphasized Variational Objective},
  booktitle={NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability},
  year={2024},
  url={https://openreview.net/forum?id=wzvP0CJ8h4},
}
```
