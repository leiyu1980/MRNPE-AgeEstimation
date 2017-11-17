import torch.nn as nn
import torch.utils.data as data
from PIL import Image
import os
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as scheduler

def mytransform():
	mytransform = transforms.Compose([transforms.ToTensor()])
	return mytransform

def default_loader(path):
    return Image.open(path).convert('RGB')

class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform = None, target_transform=None, loader=default_loader):
        fh = open(label)
        imgs=[]
        lables = []
        for line in  fh.readlines():
            cls = line.split()
            fn = cls.pop(0)
            ln = int(cls.pop(0))
            if os.path.isfile(os.path.join(root, fn)):
                #imgs.append((fn,int(ln)))
                imgs.append((fn, ln))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)