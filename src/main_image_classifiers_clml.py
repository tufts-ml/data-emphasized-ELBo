import argparse
import os
import math
import time
import copy
import types
import pandas as pd
# PyTorch
import torch
import torchvision
# Laplace
import laplace
# Importing our custom module(s)
import layers
import likelihoods
import losses
import models
import priors
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--alpha", default=0.01, help="Backbone prior precision (default: 0.01)", type=float)
    parser.add_argument("--batch_size", default=128, help="Batch size (default: 128)", type=int)
    parser.add_argument("--beta", default=0.01, help="Classifier prior precision (default: 0.01)", type=float)
    parser.add_argument("--dataset", default="", help="Dataset (default: \"\")", type=str)
    parser.add_argument("--dataset_dir", default="", help="Directory to dataset (default: \"\")", type=str)
    parser.add_argument("--experiments_dir", default="", help="Directory to save experiments (default: \"\")", type=str)
    parser.add_argument("--kappa", default=1.0, help="TODO (default: 1.0)", type=float)
    parser.add_argument("--la_batch_size", default=128, help="Batch size (default: 128)", type=int)
    parser.add_argument("--lr_0", default=0.1, help="Initial learning rate (default: 0.1)", type=float)
    parser.add_argument("--method", default="CLML", help="Method (default: \"CLML\")", type=str)
    parser.add_argument("--model", default="l2-sp", help="Model (default: \"L2-SP\")", type=str)
    parser.add_argument("--model_arch", default="ResNet-50", help="Model architecture (default: \"ResNet-50\")", type=str)
    parser.add_argument("--model_name", default="test", help="Model name (default: \"test\")", type=str)
    parser.add_argument("--n", default=1000, help="Number of training samples (default: 1000)", type=int)
    parser.add_argument("--num_workers", default=0, help="Number of workers (default: 0)", type=int)
    parser.add_argument("--prior_dir", default="", help="TODO (default: \"\")", type=str)
    parser.add_argument("--prior_type", default="", help="TODO (default: \"\")", type=str)
    parser.add_argument("--random_state", default=42, help="Random state (default: 42)", type=int)
    parser.add_argument("--save", action="store_true", default=False, help="Whether or not to save the model (default: False)")
    parser.add_argument("--temps", default=[1.0, 0.1, 0.01, 0.001, 0.0001], help="Temperatures (defaul: [1.0, 0.1, 0.01, 0.001, 0.0001])", nargs="+", type=float)
    parser.add_argument("--tune", action="store_true", default=False, help="Whether validation or test set is used (default: False)")
    args = parser.parse_args()
    
    torch.manual_seed(args.random_state)
    
    os.makedirs(args.experiments_dir, exist_ok=True)
           
    assert args.dataset in ["CIFAR-10", "Flower-102", "Pet-37"]
    if args.dataset == "CIFAR-10":
        num_classes = 10
        augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(args.dataset_dir, args.n, args.tune, args.random_state)
    elif args.dataset == "Flower-102":
        num_classes = 102
        augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_flower102_datasets(args.dataset_dir, args.n, args.tune, args.random_state)
    elif args.dataset == "Pet-37":
        num_classes = 37
        augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_pet37_datasets(args.dataset_dir, args.n, args.tune, args.random_state)

    augmented_train_dataloader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=min(args.batch_size, len(augmented_train_dataset)), shuffle=True, num_workers=args.num_workers, drop_last=True)
    la_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=min(args.la_batch_size, len(train_dataset)), num_workers=args.num_workers)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_or_test_dataloader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        
    checkpoint = torch.load(f"{args.prior_dir}/{args.prior_type}_model.pt", map_location=torch.device("cpu"), weights_only=False)
    assert args.model_arch in ["ResNet-50", "ViT-B/16", "ConvNeXt-Tiny"]
    if args.model_arch == "ResNet-50":
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        num_classifier_params = (2048 * num_classes) + num_classes
    elif args.model_arch == "ViT-B/16":
        model = torchvision.models.vit_b_16()
        model.heads = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.heads = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
        num_classifier_params = (768 * num_classes) + num_classes
    elif args.model_arch == "ConvNeXt-Tiny":
        model = torchvision.models.convnext_tiny()
        model.classifier[2] = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.classifier[2] = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
        num_classifier_params = (768 * num_classes) + num_classes
                        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
                
    assert args.model in ["l2-zero", "l2-sp"]
    if args.model == "l2-zero":
        backbone_prior_params = torch.load(f"{args.prior_dir}/{args.prior_type}_mean.pt", map_location=device, weights_only=False)
        backbone_prior_params = torch.zeros_like(backbone_prior_params)
        backbone_prior = priors.NoPrior(num_params=len(backbone_prior_params)) if args.alpha == 0.0 else priors.IsotropicGaussianPrior(num_params=len(backbone_prior_params), prior_variance=1/args.alpha)
        classifier_prior = priors.NoPrior(num_params=num_classifier_params) if args.beta == 0.0 else priors.IsotropicGaussianPrior(num_params=num_classifier_params, prior_variance=1/args.beta)
    elif args.model == "l2-sp":
        backbone_prior_params = torch.load(f"{args.prior_dir}/{args.prior_type}_mean.pt", map_location=device, weights_only=False)
        backbone_prior = priors.NoPrior(num_params=len(backbone_prior_params)) if args.alpha == 0.0 else priors.IsotropicGaussianPrior(prior_params=backbone_prior_params, prior_variance=1/args.alpha)
        classifier_prior = priors.NoPrior(num_params=num_classifier_params) if args.beta == 0.0 else priors.IsotropicGaussianPrior(num_params=num_classifier_params, prior_variance=1/args.beta)
    
    likelihood = likelihoods.CategoricalLikelihood(num_classes=num_classes)
    
    backbone_prior.to(device)
    classifier_prior.to(device)
    likelihood.to(device)
    
    assert args.method in ["CLML"]
    if args.method == "CLML":
        # NOTE: In this loss function $\alpha, \beta$ are equivalent to Li et al. (2018).
        criterion = losses.TransferLearningLoss(model, likelihood, backbone_prior, classifier_prior)
    
    optimizer = torch.optim.SGD([{"params": model.parameters()}, {"params": likelihood.parameters()}, {"params": backbone_prior.parameters()}, {"params": classifier_prior.parameters()}], lr=args.lr_0, weight_decay=0.0, momentum=0.9, nesterov=True)
    
    steps = 6000
    num_batches = len(augmented_train_dataloader)
    epochs = int(steps / num_batches)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*num_batches)
            
    for epoch in range(epochs):        
        train_epoch_start_time = time.time()
        augmented_train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_dataloader, lr_scheduler=lr_scheduler, num_samples=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_epoch_end_time = time.time()
        #train_metrics = utils.evaluate(model, criterion, train_dataloader, num_samples=1)
        train_metrics = augmented_train_metrics
        
        if args.tune or epoch == epochs - 1:
            val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_dataloader, num_samples=1)
        else:
            val_or_test_metrics = {key: 0.0 for key, value in train_metrics.items()}
            
        if epoch == epochs - 1:
            
            if args.model_arch == "ConvNeXt-Tiny":
                with torch.no_grad():
                    utils.replace_layernorm2d(model)
                    utils.replace_cnblock(model)
                    
            prior_precision = torch.tensor([args.alpha * len(train_dataset), args.beta * len(train_dataset)], device=device)
            classifier_prior_params = torch.zeros(size=(num_classifier_params,), device=device)
            prior_mean = torch.cat((backbone_prior_params, classifier_prior_params))
            la = models.L2SPLaplace(model=model, likelihood="classification", prior_precision=prior_precision, prior_mean=prior_mean, backend=laplace.curvature.AsdlEF)
            la.num_backbone_params = len(backbone_prior_params)    
            
            la.fit(la_train_dataloader)
            
            best_temp, best_metrics = max([(temp, utils.evaluate_clml(model, la, val_or_test_dataloader, num_samples=20, temp=temp)) for temp in args.temps], key=lambda x: x[1]["bma_acc"])
                                            
            val_or_test_metrics["bma_acc"] = best_metrics["bma_acc"]
            val_or_test_metrics["clml"] = best_metrics["clml"]
            val_or_test_metrics["temp"] = best_temp
            
        else:
            val_or_test_metrics["bma_acc"] = 0.0
            val_or_test_metrics["clml"] = 0.0
            val_or_test_metrics["temp"] = 0.0
          
        if epoch == 0:
            columns = ["epoch", *[f"train_{key}" for key in train_metrics.keys() if key not in ["probs", "labels"]], "train_sec/epoch", *[f"val_or_test_{key}" for key in val_or_test_metrics.keys() if key not in ["probs", "labels"]]]
            model_history_df = pd.DataFrame(columns=columns)
            
        row = [epoch, *[value for key, value in train_metrics.items() if key not in ["probs", "labels"]], train_epoch_end_time - train_epoch_start_time, *[value for key, value in val_or_test_metrics.items() if key not in ["probs", "labels"]]]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_dir}/{args.model_name}.csv')
    
    if args.save:
        torch.save(model.state_dict(), f'{args.experiments_dir}/{args.model_name}.pt')
