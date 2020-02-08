import matplotlib.pyplot as plt
import os
import torch

def enc_loss_plot(hist, path, record_iter):
    '''
    Plot a loss graph of encoder training.

    Args:
        - hist (list) : List consists of loss values.
        - path (str) : Directory to save loss graph.
        - record_iter (int) : Frequency of saving loss value in iterations. For example, if the loss value is 
                              saved in hist every ten iteration, record_iter should be ten.
    '''
    plt.switch_backend('agg')
    x = range(0, record_iter * len(hist), record_iter)
        
    plt.plot(x, hist, label='loss')
    
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, 'loss.png')
    plt.savefig(path)
    plt.close()

def cls_loss_plot(hist, path, record_epoch):
    '''
    Plot a loss graph of linear classifier training.

    Args:
        - hist (list) : List consists of loss values.
        - path (str) : Directory to save loss graph.
        - record_epoch (int) : Frequency of saving loss value in epoch. For example, if the loss value is saved
                               in hist every ten epoch, record_epoch should be ten.
    '''
    plt.switch_backend('agg')
    x = range(0, record_epoch * len(hist), record_epoch)
        
    plt.plot(x, hist, label='loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, 'loss.png')
    plt.savefig(path)
    plt.close()

def accr_plot(hist, path, record_epoch):
    '''
    Plot a accuracy graph of linear classifier.

    Args:
        - hist (list) : List consists of accuracy values.
        - path (str) : Directory to save accuracy graph.
        - record_epoch (int) : Frequency of saving accuracy value in epoch. For example, if the loss value is
                              saved in hist every ten epoch, record_epoch should be ten.
    '''
    plt.switch_backend('agg')
    x = range(0, record_epoch * len(hist), record_epoch)
    max_accr = max(hist)
    label = str(round(max_accr, 6) * 100)
    
    plt.plot(x, hist, label=label)    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, 'accuracy.png')
    plt.savefig(path)
    plt.close()