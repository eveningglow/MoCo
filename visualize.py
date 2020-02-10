# Compare
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import model
import util
from dataloader import data_loader
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()

# Config - Path
parser.add_argument('--dataset_root', type=str, default='/home/nas_datasets/sanghyeon_data/STL-10/',
                    help='Root directory of test dataset.')
parser.add_argument('--encoder_output_root', type=str, default='output',
                    help='Root directory of training results.')
parser.add_argument('--encoder_dataset_name', type=str, default='IMAGENET-64',
                    help='Name of dataset that was used to train an encoder.')
parser.add_argument('--multiple_encoder_exp_version', '--names-list', nargs='+', default=[],
                    help='Experimental versions that you want to compare')
parser.add_argument('--result_path', type=str, default='img',
                    help='Directory for saving graph plot.')

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
parser.add_argument('--load_pretrained_epoch', type=int, default=50,
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
parser.add_argument('--num_workers', type=int, default=16,
                    help='Number of workers for data loader.')

parser.add_argument('--test_num', type=int, default=3,
                    help='Number of test for a fixed encoder.')

config = parser.parse_args()

if not os.path.exists(config.result_path):
    os.makedirs(config.result_path)
    
def get_accuracy(feature_extractor, tst_dloader):
    '''
    Calculate classification accuracy
    
    Args:
        feature_extractor (net) : Pretrained feature extractor
        tst_dloader (Dataloader) : Data loader for test set
        
    Return:
        classification accuracy
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

def visualize_accuracy(accr_dict, path, record_epoch):
    '''
    Plot classification accuracy graph. It contains graph from several experimental versions to compare.
    
    Args:
        accr_dict (dict) : Classification accuracy. Each key denotes experimental version and each value contains 
                           'test_num' lists. For example, if test_num is three,
                            accr_dict['version_1] == [[accr_history_1], [accr_history_2], [accr_history_3]]
                            accr_dict['version_2] == [[accr_history_1], [accr_history_2], [accr_history_3]]
                            accr_dict['version_3] == [[accr_history_1], [accr_history_2], [accr_history_3]]

        tst_dloader (Dataloader) : Data loader for test set
        
    Return:
        classification accuracy
    '''

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    
    # Selected color
    color_names = ['steelblue', 'mediumseagreen', 'coral']

    plt.switch_backend('agg')
    plt.figure(figsize=(12, 6))
    
    total_len = len(accr_dict[list(accr_dict.keys())[0]][0])
    x = range(0, record_epoch * total_len, record_epoch)
    
    for i, (exp_version, accr_hists) in enumerate(accr_dict.items()):
        max_accr = 0
        color_val = colors[color_names[i]]

        # Find the one maximum accuracy among all tests in an experiment version
        # This value will be represented as a maximum accuracy of current experiment version
        for hist in accr_hists:
            cur_max_accr = max(hist)
            max_accr = cur_max_accr if cur_max_accr > max_accr else max_accr
            
        # plot
        for j, hist in enumerate(accr_hists):
            cur_max_accr = max(hist)
            max_accr = cur_max_accr if cur_max_accr > max_accr else max_accr
            label = 'M' + str(i+1) + ': ' + str(round(max_accr, 6) * 100) + '%'
            if j == 0:
                plt.plot(x, hist, color_val, label=label)    
            else:
                plt.plot(x, hist, color_val)    
            
    plt.xlabel('Epoch')
    plt.ylabel('Accr')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, 'accr_model_compare.png')
    plt.savefig(path)
    plt.close()

# All accuracy histories will be saved in here
# accr_of_all_model['version_1] <- [[accr_history_1], [accr_history_2], [accr_history_3]]
# accr_of_all_model['version_2] <- [[accr_history_1], [accr_history_2], [accr_history_3]]
# accr_of_all_model['version_3] <- [[accr_history_1], [accr_history_2], [accr_history_3]]
accr_of_all_model = {}
for version in config.multiple_encoder_exp_version:
    accr_of_all_model[version] = []

# For each version
for version_num, exp_version in enumerate(config.multiple_encoder_exp_version):
    # Load encoder
    encoder = nn.DataParallel(model.Resnet50(dim=config.out_dim)).to(dev)
    ckpt_name = 'ckpt_' + str(config.load_pretrained_epoch) + '.pkl'
    ckpt_path = os.path.join(config.encoder_output_root, config.encoder_dataset_name, 
                             exp_version, 'weight', ckpt_name)
    ckpt = torch.load(ckpt_path)
    encoder.load_state_dict(ckpt['encoder'])

    feature_extractor = nn.Sequential(* list(encoder.module.resnet.children())[:-1]) 
    
    # Freeze encoder
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # Loss function
    cross_entropy = nn.CrossEntropyLoss()
    
    # For each test with fixed encoder
    for t_num in range(config.test_num):
        print('[Version : %d / %d, Test : %d / %d]...' \
              %(version_num+1, len(config.multiple_encoder_exp_version), t_num+1, config.test_num))
        accr_hist = []
        
        # Build dataloader
        print('Build data loader.. \n')
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

        # linear classifier
        linear = nn.Linear(config.in_dim, config.cls_num).to(dev)
        
        # Optimizer
        optim_linear = optim.SGD(linear.parameters(),
                                 lr=config.lr,
                                 momentum=config.SGD_momentum,
                                 weight_decay=config.weight_decay)

        # Start training
        epoch = 0
        total_iters = 0

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
                
            epoch += 1
            print('[Epoch : %d / Total iters : %d] : loss : %f ...' %(epoch, total_iters, loss.item()))
                
            # Update learning rate
            factor = int(epoch / 20)
            new_lr = config.lr * (0.2**factor)

            for param_group in optim_linear.param_groups:
                print('LR is updated to %f ...' % new_lr)
                param_group['lr'] = new_lr
            
            # Calculate the current accuracy and plot the graphs
            if (epoch - 1) %config. eval_epoch == 0 or (epoch == config.max_epoch):
                linear.eval()

                accr = get_accuracy(feature_extractor, tst_dloader)
                accr_hist.append(accr)

        # Accuracy history for 100 epoch is saved in accr_of_all_model
        accr_of_all_model[exp_version].append(accr_hist)
        
visualize_accuracy(accr_of_all_model, config.result_path, record_epoch=1) 