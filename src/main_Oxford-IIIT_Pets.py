import argparse
import os
import time
import math
import random
import numpy as np
import pandas as pd
import wandb
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import losses
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')
    #TODO: alpha
    parser.add_argument('--batch_size', default=128, help='Batch size (default: 128)', type=int)
    #TODO: beta
    parser.add_argument('--dataset_directory', default='', help='Directory to dataset (default: \'\')', type=str)
    parser.add_argument('--experiments_directory', default='', help='Directory to save experiments (default: \'\')', type=str)
    parser.add_argument('--K', default=5, help='Rank of low-rank covariance matrix (default: 5)', type=int)
    parser.add_argument('--kappa', default=1.0, help='TODO (default: 1.0)', type=float)
    parser.add_argument('--lr_0', default=0.5, help='Initial learning rate (default: 0.5)', type=float)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
    parser.add_argument('--N', default=1000, help='Number of training samples (default: 1000)', type=int)
    parser.add_argument('--num_workers', default=0, help='Number of workers (default: 0)', type=int)
    parser.add_argument('--prior_eps', default=0.1, help='Added to prior variance (default: 0.1)', type=float) # Default from "Pre-Train Your Loss"
    parser.add_argument('--prior_directory', help='Directory to priors (default: \'\')', type=str)
    parser.add_argument('--prior_type', help='TODO', type=str)
    parser.add_argument('--save', action='store_true', default=False, help='Whether or not to save the model (default: False)')
    parser.add_argument('--tune', action='store_true', default=False, help='Whether validation or test set is used (default: False)')
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--wandb', action='store_true', default=False, help='Whether or not to log to Weights & Biases')
    parser.add_argument('--wandb_project', default='test', help='Weights & Biases project name (default: \'test\')', type=str)
    args = parser.parse_args()
    
    if args.wandb:
        # TODO: Add prior_type to wandb config
        wandb.login()
        os.environ['WANDB_API_KEY'] = '4bfaad8bea054341b5e8729c940150221bdfbb6c'
        wandb.init(
            project = args.wandb_project,
            name = args.model_name,
            config={
                'batch_size': args.batch_size,
                'dataset_directory': args.dataset_directory,
                'experiments_directory': args.experiments_directory,
                'K': args.K,
                'kappa': args.kappa,
                'lr_0': args.lr_0,
                'model_name': args.model_name,
                'N': args.N,
                'num_workers': args.num_workers,
                'prior_eps': args.prior_eps,
                'prior_directory': args.prior_directory,
                'prior_type': args.prior_type,
                'save': args.save,
                'tune': args.tune,
                'random_state': args.random_state,
                'wandb': args.wandb,
                'wandb_project': args.wandb_project,                
            }
        )
    
    torch.manual_seed(args.random_state)
    utils.makedir_if_not_exist(args.experiments_directory)
           
    augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_oxfordiiit_pets_datasets(root=args.dataset_directory, n=args.N, tune=args.tune, random_state=args.random_state)

    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=min(args.batch_size, len(augmented_train_dataset)), shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), num_workers=args.num_workers)
    val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    num_classes = 37
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    checkpoint = torch.load(f'{args.prior_directory}/resnet50_torchvision_model.pt', map_location=torch.device('cpu'))
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Identity()
    model.load_state_dict(checkpoint)
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    model.to(device)

    #criterion = losses.L2SPLoss(args.alpha, bb_loc, args.beta)
    if args.prior_type == 'l2-zero':
        model.sigma_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1e-4, device=device))))
        utils.add_variational_layers(model, model.sigma_param)
        bb_loc = torch.load(f'{args.prior_directory}/resnet50_torchvision_mean.pt', map_location=torch.device('cpu')).to(device)
        bb_loc = torch.zeros_like(bb_loc).to(device)
        criterion = losses.L2KappaELBOLoss(bb_loc, args.kappa, model.sigma_param)
    elif args.prior_type == 'l2-sp':
        model.sigma_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1e-4, device=device))))
        utils.add_variational_layers(model, model.sigma_param)
        bb_loc = torch.load(f'{args.prior_directory}/resnet50_torchvision_mean.pt', map_location=torch.device('cpu')).to(device)
        criterion = losses.L2KappaELBOLoss(bb_loc, args.kappa, model.sigma_param)
    elif args.prior_type == 'ptyl':
        model.sigma_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1e-4, device=device))))
        utils.add_variational_layers(model, model.sigma_param)
        bb_loc = torch.load(f'{args.prior_directory}/resnet50_torchvision_mean.pt', map_location=torch.device('cpu')).to(device)
        Sigma_diag = torch.load(f'{args.prior_directory}/resnet50_torchvision_variance.pt', map_location=torch.device('cpu')).to(device)
        Q = torch.load(f'{args.prior_directory}/resnet50_torchvision_covmat.pt', map_location=torch.device('cpu')).to(device).T
        criterion = losses.PTYLKappaELBOLoss(bb_loc, args.kappa, Q, Sigma_diag, model.sigma_param, K=args.K, prior_eps=args.prior_eps)
    else:
        raise NotImplementedError(f'The specified prior type \'{args.prior_type}\' is not implemented.')

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, weight_decay=0.0, momentum=0.9, nesterov=True)

    steps = 6000
    num_batches = len(augmented_train_loader)
    epochs = int(steps/num_batches)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*num_batches)
    
    columns = ['epoch', 'sigma', 'train_acc', 'train_loss', 'train_nll', 'train_sec/epoch', 'val_or_test_acc', 'val_or_test_loss', 'val_or_test_nll']
    model_history_df = pd.DataFrame(columns=columns)
            
    for epoch in range(epochs):
        start_time = time.time()
        augmented_train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_loader, lr_scheduler=lr_scheduler, num_classes=num_classes)
        end_time = time.time()
        train_metrics = augmented_train_metrics
        val_or_test_metrics = {'acc': 0.0, 'loss': 0.0, 'nll': 0.0}
        
        if epoch == epochs-1:
            num_samples = 10
            sample_metrics = [utils.evaluate(model, criterion, train_loader, num_classes=num_classes, use_posterior=True) for _ in range(num_samples)]
            train_metrics = {key: sum(metrics[key] for metrics in sample_metrics) / num_samples for key in ['acc', 'loss', 'nll']}
            val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_loader, num_classes=num_classes)
                            
        row = [epoch, torch.nn.functional.softplus(model.sigma_param).item(), train_metrics['acc'], train_metrics['loss'], train_metrics['nll'], end_time-start_time, val_or_test_metrics['acc'], val_or_test_metrics['loss'], val_or_test_metrics['nll']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        if args.wandb:
            wandb.log({
                'epoch': epoch, 
                'sigma': torch.nn.functional.softplus(model.sigma_param).item(),
                'train_acc': train_metrics['acc'], 
                'train_loss': train_metrics['loss'], 
                'train_nll': train_metrics['nll'], 
                'val_or_test_acc': val_or_test_metrics['acc'], 
                'val_or_test_loss': val_or_test_metrics['loss'], 
                'val_or_test_nll': val_or_test_metrics['nll'], 
            })
        
        model_history_df.to_csv(f'{args.experiments_directory}/{args.model_name}.csv')
    
    if args.save:
        torch.save(model.state_dict(), f'{args.experiments_directory}/{args.model_name}.pt')
        