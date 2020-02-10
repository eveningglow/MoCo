import os
import sys
from PIL import Image
import random

import torchvision.datasets as Datasets
import torchvision.transforms as Transforms
from torchvision.datasets import VisionDataset, ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

''' 
Codes between ============= <- these lines are copied from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py with small changes.
Changed part is commented with 'CHANGED'.
'''

''' =========================================================================================== '''
def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class DatasetFolder(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    ''' CHANGED : Return two tensors randomly augmented from the same image '''
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            aug_sample_1 = self.transform(sample)
            aug_sample_2 = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return aug_sample_1, aug_sample_2, target

    def __len__(self):
        return len(self.samples)
    
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
class AugImageFolder(DatasetFolder):
    '''
    It is similar to ImageFolder but samples two tensors randomly augmented from the same images.

    Args: 
        Same with ImageFolder
    '''

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(AugImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                             transform=transform,
                                             target_transform=target_transform,
                                             is_valid_file=is_valid_file)
        self.imgs = self.samples
''' =========================================================================================== '''


def data_loader(dataset_root='/home/nas_datasets/ILSVRC/Data/CLS-LOC/train', resize=84, crop=64, 
                batch_size=64, num_workers=16, type='train'):    
    '''
    Data loader for MoCo. It is written assuming that 'ImageNet' dataset is used to train an encoder in 
    self-supervised manner, and 'STL-10' dataset is used to evaluate the encoder.

    Args:
        - dataset_root (str): Root directory consisting of subdirectories for each class. Each subdirectory 
                              contains images corresponding that specific class. Note that the class label
                              is not used in training, but this constraint is caused by the structure of 
                              Imagenet dataset.
        - resize (int) : Images are resized with this value.
        - crop (int) : Images are cropped with this value. This is a final size of image transformation.
        - batch_size (int) : Batch size
        - num_workers (int) : Number of workers for data loader
        - type (str) : Type of data loader.
                       1) encoder_train : data loader for training an encoder in self-supervised manner.
                       2) classifier_train : data loader for training a linear classifier to evaluate 
                                             the encoder.
                       3) classifier_test : data loader for evaluating the linear classifier.
                       
    Returns:
        - dloader : Data loader
        - dlen : Total number of data
    '''

    transform_list = []
    if type == 'encoder_train':
        transform_list += [Transforms.RandomResizedCrop(size=crop),
                           Transforms.ColorJitter(0.1, 0.1, 0.1),
                           Transforms.RandomHorizontalFlip(),
                           Transforms.RandomGrayscale()]
    elif type == 'classifier_train':
        transform_list += [Transforms.Resize(size=resize),
                           Transforms.RandomCrop(size=crop),
                           Transforms.RandomHorizontalFlip()]
    elif type == 'classifier_test':
        transform_list += [Transforms.Resize(size=resize),
                           Transforms.CenterCrop(size=crop)]

    transform_list += [Transforms.ToTensor(),
                       Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                            std=(0.5, 0.5, 0.5))]

    transform = Transforms.Compose(transform_list)
    
    if type == 'encoder_train':
        dset = AugImageFolder(root=dataset_root, transform=transform)
    elif type == 'classifier_train' or type == 'classifier_test':
        split = type.split('_')[-1] # 'train' or 'test'
        dset = Datasets.STL10(root=dataset_root, split=split, transform=transform, download=True)
        
    dlen = len(dset)
    dloader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dloader, dlen