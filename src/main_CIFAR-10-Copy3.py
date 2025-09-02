import argparse
import time
import types
import pandas as pd
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import layers
import losses
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--alpha', default=0.01, help='Backbone weight decay (default: 0.01)', type=float)
    parser.add_argument('--batch_size', default=128, help='Batch size (default: 128)', type=int)
    parser.add_argument('--beta', default=0.01, help='Classifier head weight decay (default: 0.01)', type=float)
    parser.add_argument('--criterion', default='l2-sp', help='Criterion (default: \'l2-sp\')', type=str)
    parser.add_argument('--dataset_directory', default='', help='Directory to dataset (default: \'\')', type=str)
    parser.add_argument('--ELBo', action='store_true', default=False, help='Whether or not to learn regularization strength with the ELBo (default: False)')
    parser.add_argument('--experiments_directory', default='', help='Directory to save experiments (default: \'\')', type=str)
    parser.add_argument('--kappa', default=1.0, help='TODO (default: 1.0)', type=float)
    parser.add_argument('--lambd', default=1.0, help='Covariance scaling factor (default: 1.0)', type=float)
    parser.add_argument('--lr_0', default=0.1, help='Initial learning rate (default: 0.1)', type=float)
    parser.add_argument('--model_arch', default='ResNet-50', help='Model architecture (default: \'ResNet-50\')', type=str)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
    parser.add_argument('--n', default=1000, help='Number of training samples (default: 1000)', type=int)
    parser.add_argument('--num_workers', default=0, help='Number of workers (default: 0)', type=int)
    parser.add_argument('--prior_directory', default='/cluster/tufts/hugheslab/eharve06/resnet50_torchvision', help='TODO (default: \'/cluster/tufts/hugheslab/eharve06/resnet50_torchvision\')', type=str)
    parser.add_argument('--prior_type', default='resnet50_torchvision', help='TODO (default: \'resnet50_torchvision\')', type=str)
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--save', action='store_true', default=False, help='Whether or not to save the model (default: False)')
    parser.add_argument('--tune', action='store_true', default=False, help='Whether validation or test set is used (default: False)')
    args = parser.parse_args()
    
    torch.manual_seed(args.random_state)
    utils.makedir_if_not_exist(args.experiments_directory)
           
    augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(args.dataset_directory, args.n, args.tune, args.random_state)

    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=min(args.batch_size, len(augmented_train_dataset)), shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), num_workers=args.num_workers)
    val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    num_classes = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
        
    checkpoint = torch.load(f'{args.prior_directory}/{args.prior_type}_model.pt', map_location=torch.device('cpu'), weights_only=False)
    if args.model_arch == 'ResNet-50':
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif args.model_arch == 'ViT-B/16':
        model = torchvision.models.vit_b_16()
        model.heads = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.heads = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
    elif args.model_arch == 'ConvNeXt_Tiny':
        model = torchvision.models.convnext_tiny()
        model.classifier[2] = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.classifier[2] = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
    else:
        raise NotImplementedError(f'The specified model architecture \'{args.model_arch}\' is not implemented.')
    model.to(device)
    
    if args.criterion == 'l2-zero' and not args.ELBo:
        criterion = losses.L2ZeroLoss(args.alpha)
    elif args.criterion == 'l2-zero' and args.ELBo:        
        model.sigma_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1e-4, device=device))))
        utils.add_variational_layers(model, model.sigma_param)
        model.use_posterior = types.MethodType(utils.use_posterior, model)
        #bb_loc = torch.load(f'{args.prior_directory}/{args.prior_type}_mean.pt', map_location=torch.device('cpu'), weights_only=False).to(device)
        #bb_loc = torch.zeros_like(bb_loc).to(device)
        criterion = losses.L2ZeroKappaELBoLoss(args.kappa, model.sigma_param)
    elif args.criterion == 'l2-sp' and not args.ELBo:
        bb_loc = torch.load(f'{args.prior_directory}/{args.prior_type}_mean.pt', map_location=torch.device('cpu'), weights_only=False).to(device)
        criterion = losses.L2SPLoss(args.alpha, bb_loc, args.beta)
    elif args.criterion == 'l2-sp' and args.ELBo:
        model.sigma_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1e-4, device=device))))
        utils.add_variational_layers(model, model.sigma_param)
        model.use_posterior = types.MethodType(utils.use_posterior, model)
        bb_loc = torch.load(f'{args.prior_directory}/{args.prior_type}_mean.pt', map_location=torch.device('cpu'), weights_only=False).to(device)
        criterion = losses.L2KappaELBoLoss(bb_loc, args.kappa, model.sigma_param)
    elif args.criterion == 'ptyl' and not args.ELBo:
        bb_loc = torch.load(f'{args.prior_directory}/{args.prior_type}_mean.pt', map_location=torch.device('cpu'), weights_only=False).to(device)
        # Note: Released covmat is shape K \times D. PTYLLoss() expects Q to be shape D \times K.
        Q = torch.load(f'{args.prior_directory}/{args.prior_type}_covmat.pt', map_location=torch.device('cpu'), weights_only=False).to(device).T
        Sigma_diag = torch.load(f'{args.prior_directory}/{args.prior_type}_variance.pt', map_location=torch.device('cpu'), weights_only=False).to(device)
        criterion = losses.PTYLLoss(bb_loc, args.beta, args.lambd, Q, Sigma_diag)
    elif args.criterion == 'ptyl' and args.ELBo:
        model.sigma_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1e-4, device=device))))
        utils.add_variational_layers(model, model.sigma_param)
        model.use_posterior = types.MethodType(utils.use_posterior, model)
        bb_loc = torch.load(f'{args.prior_directory}/{args.prior_type}_mean.pt', map_location=torch.device('cpu'), weights_only=False).to(device)
        # Note: Released covmat is shape K \times D. PTYLKappaELBoLoss() expects Q to be shape D \times K.
        Q = torch.load(f'{args.prior_directory}/{args.prior_type}_covmat.pt', map_location=torch.device('cpu'), weights_only=False).to(device).T
        Sigma_diag = torch.load(f'{args.prior_directory}/{args.prior_type}_variance.pt', map_location=torch.device('cpu'), weights_only=False).to(device)
        criterion = losses.PTYLKappaELBoLoss(bb_loc, args.kappa, Q, Sigma_diag, model.sigma_param)
    else:
        raise NotImplementedError(f'The specified criterion \'{args.criterion}\' is not implemented.')
        
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, weight_decay=0.0, momentum=0.9, nesterov=True)
    
    # TODO: Added for linear probing
    #for param in model.parameters():
    #    param.requires_grad = False
    #model.fc.weight.requires_grad = True
    #model.fc.bias.requires_grad = True
    #optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr_0, weight_decay=0.0, momentum=0.9, nesterov=True)

    steps = 6000
    num_batches = len(augmented_train_loader)
    epochs = int(steps/num_batches)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*num_batches)
    
    if not args.ELBo:
        columns = ['epoch', 'train_acc', 'train_loss', 'train_nll', 'train_sec/epoch', 'val_or_test_acc', 'val_or_test_loss', 'val_or_test_nll']
    else:
        #columns = ['epoch', 'lambda_star', 'sigma', 'tau_star', 'train_acc', 'train_loss', 'train_nll', 'train_sec/epoch', 'val_or_test_acc', 'val_or_test_loss', 'val_or_test_nll']
        columns = ['epoch', 'lambda_star', 'sigma', 'tau_star', 'train_acc', 'train_kl', 'train_loss', 'train_nll', 'train_sec/epoch', 'train_eval_acc', 'train_eval_kl', 'train_eval_loss', 'train_eval_nll', 'val_or_test_acc', 'val_or_test_kl', 'val_or_test_loss', 'val_or_test_nll'] #
        #columns = ['epoch', 'lambda_star', 'sigma', 'train_acc', 'train_loss', 'train_nll', 'train_sec/epoch', 'val_or_test_acc', 'val_or_test_loss', 'val_or_test_nll']

    model_history_df = pd.DataFrame(columns=columns)
    
    for epoch in range(epochs):
        train_epoch_start_time = time.time()
        augmented_train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_loader, lr_scheduler=lr_scheduler, num_classes=num_classes)
        train_epoch_end_time = time.time()
        #train_metrics = utils.evaluate(model, criterion, train_loader, num_classes=num_classes)
        train_metrics = augmented_train_metrics
        
        if args.tune or epoch == epochs-1:
            if not args.ELBo:
                val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_loader, num_classes=num_classes)
            else:
                model.use_posterior(True)
                num_samples = 10
                sample_metrics = [utils.evaluate(model, criterion, train_loader, num_classes=num_classes) for _ in range(num_samples)]
                train_metrics = {key: sum(metrics[key] for metrics in sample_metrics) / num_samples for key in ['acc', 'kl', 'loss', 'nll']}
                model.use_posterior(False)
                val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_loader, num_classes=num_classes)
        else:
            val_or_test_metrics = {'acc': 0.0, 'kl': 0.0, 'loss': 0.0, 'nll': 0.0}
            model.use_posterior(True) #
            train_metrics = utils.evaluate(model, criterion, train_loader, num_classes=num_classes) #
            model.use_posterior(False) #
            val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_loader, num_classes=num_classes) #
            
                
        if not args.ELBo:
            row = [epoch, train_metrics['acc'], train_metrics['loss'], train_metrics['nll'], train_epoch_end_time-train_epoch_start_time, val_or_test_metrics['acc'], val_or_test_metrics['loss'], val_or_test_metrics['nll']]
        else:
            #row = [epoch, augmented_train_metrics['lambda_star'].item(), torch.nn.functional.softplus(model.sigma_param).item(), augmented_train_metrics['tau_star'].item(), train_metrics['acc'], train_metrics['loss'], train_metrics['nll'], train_epoch_end_time-train_epoch_start_time, val_or_test_metrics['acc'], val_or_test_metrics['loss'], val_or_test_metrics['nll']]
            row = [epoch, augmented_train_metrics['lambda_star'].item(), torch.nn.functional.softplus(model.sigma_param).item(), augmented_train_metrics['tau_star'].item(), augmented_train_metrics['acc'], augmented_train_metrics['kl'], augmented_train_metrics['loss'], augmented_train_metrics['nll'], train_epoch_end_time-train_epoch_start_time, train_metrics['acc'], train_metrics['kl'], train_metrics['loss'], train_metrics['nll'], val_or_test_metrics['acc'], val_or_test_metrics['kl'], val_or_test_metrics['loss'], val_or_test_metrics['nll']] #
            #row = [epoch, augmented_train_metrics['lambda_star'].item(), torch.nn.functional.softplus(model.sigma_param).item(), train_metrics['acc'], train_metrics['loss'], train_metrics['nll'], train_epoch_end_time-train_epoch_start_time, val_or_test_metrics['acc'], val_or_test_metrics['loss'], val_or_test_metrics['nll']]
            
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_directory}/{args.model_name}.csv')
    
    if args.save:
        torch.save(model.state_dict(), f'{args.experiments_directory}/{args.model_name}.pt')
