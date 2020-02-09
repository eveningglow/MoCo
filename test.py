import os
import argparse
import pickle
import pprint

import torch
import torch.nn as nn
import torch.optim as optim

import model
import util
from dataloader import data_loader

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


''' ######################## < Step 1 > Parsing test arguments ######################## '''

parser = argparse.ArgumentParser()

# Config - Path
parser.add_argument('--dataset_root', type=str, default='/home/nas_datasets/sanghyeon_data/STL-10/',
                    help='Root directory of test dataset.')
parser.add_argument('--encoder_output_root', type=str, default='output',
                    help='Root directory of training results.')
parser.add_argument('--encoder_dataset_name', type=str, default='IMAGENET-64',
                    help='Name of dataset that was used to train an encoder.')
parser.add_argument('--encoder_exp_version', type=str, default='v1',
                    help='Version of experiment.')

# Config - Hyperparameter
parser.add_argument('--trn_batch_size', type=int, default=256,
                    help='Batch size to train a linear classifier.')
parser.add_argument('--tst_batch_size', type=int, default=100,
                    help='Batch size to evaluate a linear classifier.')
parser.add_argument('--lr', type=float, default=10,
                    help='Learning rate to train a linear classifier.')
parser.add_argument('--SGD_momentum', type=float, default=0.9,
                    help='Momentum of SGD optimizer to train a linear classifier.')
parser.add_argument('--weight_decay', type=float, default=0,
                     help='Weight of L2 regularization of SGD optimizer.')

# Config - Architecture
parser.add_argument('--out_dim', type=int, default=128,
                    help='Output dimension of a last fully connected layer in encoder.')
parser.add_argument('--in_dim', type=int, default=2048,
                    help='Intput dimension of a last fully connected layer in encoder or a linear classifier.')

# Config - Setting
parser.add_argument('--load_pretrained_epoch', type=int, default=100,
                    help='Epoch to load the weight of pretrained encoder.')
parser.add_argument('--cls_num', type=int, default=10,
                    help='Number of test dataset classes.')
parser.add_argument('--resize', type=int, default=84,
                    help='Image is resized to this value.')
parser.add_argument('--crop', type=int, default=64,
                    help='Image is cropped to this value. This is the final size of image transformation.')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='Maximum epoch to train an encoder.')
parser.add_argument('--eval_epoch', type=int, default=2,
                    help='Frequency of evaluate an encoder.')
parser.add_argument('--save_weight_epoch', type=int, default=10,
                    help='Frequency of saving weight.')
parser.add_argument('--num_workers', type=int, default=16,
                    help='Number of workers for data loader.')

config = parser.parse_args()

# Show config
print('\n======================= Training configuration =======================\n')
pprint.pprint(vars(config))
print('\n======================================================================\n')

# Make output directories
loss_path = os.path.join(config.encoder_output_root, config.encoder_dataset_name, config.encoder_exp_version, 'eval/loss')
accr_path = os.path.join(config.encoder_output_root, config.encoder_dataset_name, config.encoder_exp_version, 'eval/accr')
weight_path = os.path.join(config.encoder_output_root, config.encoder_dataset_name, config.encoder_exp_version, 'eval/weight')

if not os.path.exists(loss_path):
    os.makedirs(loss_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)
if not os.path.exists(accr_path):
    os.makedirs(accr_path)

    
''' ######################## < Step 2 > Create instances ######################## '''

# Build dataloader
print('[1 / 2]. Build data loader.. \n')
trn_dloader, trn_dlen = data_loader(dataset_root=config.dataset_root,
                                    resize=config.resize, 
                                    crop=config.crop,
                                    batch_size=config.trn_batch_size,
                                    num_workers=config.num_workers,
                                    type='classifier_train')

tst_dloader, tst_dlen = data_loader(dataset_root=config.dataset_root,
                                    resize=config.resize, 
                                    crop=config.crop,
                                    batch_size=config.tst_batch_size,
                                    num_workers=config.num_workers,
                                    type='classifier_test')

# Build models
print('[2 / 2]. Build models.. \n')
encoder = nn.DataParallel(model.Resnet50(dim=config.out_dim)).to(dev)

ckpt_name = 'ckpt_' + str(config.load_pretrained_epoch) + '.pkl'
ckpt_path = os.path.join(config.encoder_output_root, config.encoder_dataset_name, config.encoder_exp_version, 'weight', ckpt_name)
ckpt = torch.load(ckpt_path)
encoder.load_state_dict(ckpt['encoder'])

feature_extractor = nn.Sequential(* list(encoder.module.resnet.children())[:-1]) # feature extractor from encoder
linear = nn.Linear(config.in_dim, config.cls_num).to(dev) # linear classifier

# Freeze encoder
for param in feature_extractor.parameters():
    param.requires_grad = False

# Optimizer
optim_linear = optim.SGD(linear.parameters(),
                         lr=config.lr,
                         momentum=config.SGD_momentum,
                         weight_decay=config.weight_decay)
# Loss function
cross_entropy = nn.CrossEntropyLoss()

# Status
loss_hist = []
accr_hist = []


''' ######################## < Step 3 > Define methods ######################## '''

def get_accuracy():
    '''
    Calculate classification accuracy
    '''
    
    total_num = 0
    correct_num = 0
    print('\n Calculate accuracy ...')
    
    with torch.no_grad():
        for idx, (img, label) in enumerate(tst_dloader):
            if idx % 50 == 0:
                print('    [%d / %d] ... ' % (idx, int(tst_dlen / config.tst_batch_size)))

            img = img.to(dev)
            label = label.to(dev)
            feature = feature_extractor(img)
            score = linear(feature.view(feature.size(0), feature.size(1)))
            pred = torch.argmax(score, dim=1, keepdim=True)

            total_num = total_num + img.size(0)
            correct_num = correct_num + (label.unsqueeze(dim=1) == pred).sum().item()
        print()
    return correct_num / total_num

def update_lr(epoch):    
    '''
    Learning rate scheduling.

    Args:
        epoch (float): Set new learning rate by a given epoch.
    '''

    # Decay 0.1 times every 20 epoch.
    factor = int(epoch / 20)
    lr = config.lr * (0.2**factor)

    for param_group in optim_linear.param_groups:
        print('LR is updated to %f ...' % lr)
        param_group['lr'] = lr
            
            
''' ######################## < Step 4 > Start training ######################## '''

epoch = 0
total_iters = 0

# Train a linear classifier
while(epoch < config.max_epoch):
    for i, (img, label) in enumerate(trn_dloader):
        # Preprocess
        linear.train()
        optim_linear.zero_grad()
        
        # Forward prop
        img = img.to(dev)
        label = label.to(dev)
        
        feature = feature_extractor(img).detach()
        score = linear(feature.view(feature.size(0), feature.size(1)))
        loss = cross_entropy(score, label)
        
        # Back prop
        loss.backward()
        optim_linear.step()
        
        # Print training status and save log
        total_iters += 1
        print('[Epoch : %d / Total iters : %d] : loss : %f ...' %(epoch, total_iters, loss.item()))
    
    epoch += 1
    
    # Update learning rate
    update_lr(epoch)
    
    # Save loss value
    loss_hist.append(loss.item())
    
    # Calculate the current accuracy and plot the graphs
    if (epoch - 1) % config.eval_epoch == 0:
        linear.eval()

        accr = get_accuracy()
        accr_hist.append(accr)
        print('[Epoch : %d / Total iters : %d] : accr : %f ...' %(epoch, total_iters, accr))

        util.cls_loss_plot(loss_hist, loss_path, record_epoch=1)
        util.accr_plot(accr_hist, accr_path, record_epoch=config.eval_epoch)
            
    # Save
    if (epoch - 1) % config.save_weight_epoch == 0:
        path_ckpt = os.path.join(weight_path, 'ckpt_' + str(epoch-1) + '.pkl')
        ckpt = linear.state_dict()
        torch.save(ckpt, path_ckpt)
        
        with open(os.path.join(loss_path, 'loss.pkl'), 'wb') as f:
            pickle.dump(loss_hist, f)
            
        with open(os.path.join(accr_path, 'accr.pkl'), 'wb') as f:
            pickle.dump(accr_hist, f)