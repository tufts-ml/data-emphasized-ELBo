import argparse
import time
import types
import pandas as pd
# PyTorch
import torch
import torchvision
# Laplace
from laplace.baselaplace import DiagLaplace
from laplace.curvature import AsdlEF
# Importing our custom module(s)
import layers
import losses
import utils

class L2SPLaplace(DiagLaplace):
    
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
        alpha = self.prior_precision[0]
        beta = self.prior_precision[1]
        return torch.cat([alpha * torch.ones(self.D, device=self._device, dtype=self._dtype),
                          beta * torch.ones(self.n_params - self.D, device=self._device, dtype=self._dtype)])

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--alpha', default=0.01, help='Backbone weight decay (default: 0.01)', type=float)
    parser.add_argument('--batch_size', default=128, help='Batch size (default: 128)', type=int)
    parser.add_argument('--beta', default=0.01, help='Classifier head weight decay (default: 0.01)', type=float)
    parser.add_argument('--criterion', default='l2-sp', help='Criterion (default: \'l2-sp\')', type=str)
    parser.add_argument('--dataset_directory', default='', help='Directory to dataset (default: \'\')', type=str)
    parser.add_argument('--experiments_directory', default='', help='Directory to save experiments (default: \'\')', type=str)
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
    parser.add_argument('--substring', default='fc', help='Substring (default: \'fc\')', type=str)
    parser.add_argument('--tune', action='store_true', default=False, help='Whether validation or test set is used (default: False)')
    args = parser.parse_args()
    
    torch.manual_seed(args.random_state)
    utils.makedir_if_not_exist(args.experiments_directory)
           
    augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(args.dataset_directory, args.n, args.tune, args.random_state)
    #augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_oxfordiiit_pet_datasets(args.dataset_directory, args.n, args.tune, args.random_state)
    #augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_flowers_102_datasets(args.dataset_directory, args.n, args.tune, args.random_state)

    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=min(args.batch_size, len(augmented_train_dataset)), shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=min(args.batch_size, len(train_dataset)), num_workers=args.num_workers)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), num_workers=args.num_workers)
    val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    num_classes = 10
    #num_classes = 37
    #num_classes = 102
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
        model.heads.head = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.heads.head = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
    elif args.model_arch == 'ConvNeXt_Tiny':
        model = torchvision.models.convnext_tiny()
        model.classifier[2] = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.classifier[2] = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
    else:
        raise NotImplementedError(f'The specified model architecture \'{args.model_arch}\' is not implemented.')
    model.to(device)
    
    prior_precision = torch.tensor([1.0, 1.0]).to(device)
    bb_loc = torch.load(f'{args.prior_directory}/{args.prior_type}_mean.pt', map_location=torch.device('cpu'), weights_only=False).to(device)   
    clf_loc = torch.zeros((2048 * num_classes) + num_classes).to(device)
    prior_mean = torch.cat([bb_loc, clf_loc])
    la = L2SPLaplace(model=model, likelihood='classification', prior_precision=prior_precision, prior_mean=prior_mean, backend=AsdlEF)
    la.D = len(bb_loc)
    alpha = la.prior_precision[0].item() / len(train_loader.dataset)
    beta = la.prior_precision[1].item() / len(train_loader.dataset)
    bb_loc = torch.zeros_like(bb_loc)
    criterion = losses.L2SPLoss(alpha, bb_loc, beta)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, weight_decay=0.0, momentum=0.9, nesterov=True)
    
    steps = 6000
    num_batches = len(augmented_train_loader)
    epochs = int(steps/num_batches)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*num_batches)
    
    F, K, gamma = 1, 100, 1
    
    columns = ['epoch', 'alpha', 'beta', 'train_acc', 'train_loss', 'train_log_marglik', 'train_nll', 'train_sec/epoch', 'val_or_test_acc', 'val_or_test_loss', 'val_or_test_nll']
    model_history_df = pd.DataFrame(columns=columns)
    
    for epoch in range(epochs):
        train_epoch_start_time = time.time()
        
        augmented_train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_loader, lr_scheduler=lr_scheduler, num_classes=num_classes)
        
        if epoch % F == 0:
            la.fit(train_loader)
            la.optimize_prior_precision(pred_type='glm', method='marglik', n_steps=K, lr=gamma, init_prior_prec=la.prior_precision)
            criterion.alpha = la.prior_precision[0].item() / len(train_loader.dataset)
            criterion.beta = la.prior_precision[1].item() / len(train_loader.dataset)
                
        train_epoch_end_time = time.time()
        #train_metrics = utils.evaluate(model, criterion, train_loader, num_classes=num_classes)
        train_metrics = augmented_train_metrics
        
        if args.tune or epoch == epochs-1:
            val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_loader, num_classes=num_classes)
        else:
            val_or_test_metrics = {'acc': 0.0, 'loss': 0.0, 'nll': 0.0}
        
        row = [epoch, criterion.alpha, criterion.beta, train_metrics['acc'], train_metrics['loss'], la.log_marginal_likelihood().item(), train_metrics['nll'], train_epoch_end_time-train_epoch_start_time, val_or_test_metrics['acc'], val_or_test_metrics['loss'], val_or_test_metrics['nll']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_directory}/{args.model_name}.csv')
    
    if args.save:
        torch.save(model.state_dict(), f'{args.experiments_directory}/{args.model_name}.pt')
