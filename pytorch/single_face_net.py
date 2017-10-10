#Lack of LRN layer, resulting 0.1~0.2 MAE error.

import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as scheduler
import os
import torch
import torch.utils.data as data
from PIL import Image
import dataloader

#torch.cuda.manual_seed_all(1)
fsave = open('accuracy3.txt','w')
batch_size = 64
test_batch_size = 100
epochs = 12
lr = 0.001
momentum = 0.9
weight_decay = 0.0005
stepsize = 5
log_interval = 100
gamma = 0.1
test_interval = 500
outf = "./snapshot3/"
root = "/fast/age/"
train_label = "/fast/age/224dataset/s1_train/Face_train_s.txt"
val_label = "/fast/age/224dataset/s1_val/face_val.txt"


def weights_init(m):
    global first_fc
    classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.normal(m.weight.data,std=0.01)
    #     nn.init.constant(m.bias.data,0.1)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('fc1') != -1:
    #         nn.init.normal(m.weight.data,std=0.005)
    #         nn.init.constant(m.bias.data,0.1)
    # elif classname.find('fc2') != -1:
    #         nn.init.normal(m.weight.data,std=0.01)
    #         nn.init.constant(m.bias.data,0.1)



class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            #nn.LRN()  Waitting for pytorch's support
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode = True),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            #nn.LRN()  Waitting for pytorch's support
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode = True),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.Conv2d(384, 384, kernel_size=3, padding=1,groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode = True),
        )
        fc1 =  nn.Linear(256 * 6 * 6, 64)
        fc2 = nn.Linear(64, 101)
        self.classifier = nn.Sequential(
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc2,
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

train_set  = dataloader.myImageFloder(root = root, label = train_label, transform = dataloader.mytransform())
trainloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=False, num_workers=4)
test_set  = dataloader.myImageFloder(root = root, label = val_label , transform = dataloader.mytransform())
testloader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size, shuffle=False, num_workers=2)
model = AlexNet()
model.apply(weights_init)
model.cuda()
params = []
for name, value in model.named_parameters():
   if 'bias' in name:
       params += [{'params':[value],'lr':lr ,'weight_decay':weight_decay}]
   else:
       params += [{'params':[value],'lr':2*lr ,'weight_decay':0}]
optimizer = optim.SGD( params, momentum=momentum)
scheduler = scheduler.StepLR(optimizer, step_size=stepsize, gamma = gamma)

age_range = []
for i in range (0,101):
    age_range.append(i)
age_range = Variable(torch.FloatTensor(age_range).cuda())

def test(batch_idx = 0, save=False):
    model.eval()
    test_loss = 0
    MAE = 0
    avg_MAE = 0
    for data, target in testloader:
        data, target = data.cuda(), torch.LongTensor(target).cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        output = F.softmax(output)
        pred = torch.sum((age_range * output),1)
        MAE += torch.sum(torch.abs(pred - target.float()),0).data
    test_loss /= len(testloader.dataset)
    avg_MAE = MAE[0] / len(testloader.dataset) 
    print('\nTest set: Average Test loss: {:.4f}, Average MAE: {:.4f}, Total MAE: {}, Batch_idx: {}\n'.format(
        test_loss, avg_MAE, MAE[0], batch_idx))
    if save == True:
        print >> fsave,'Test Test set: Average loss: {:.4f}, Average MAE: {:.4f}, Total MAE: {}, Batch_idx: {}\n'.format(test_loss, avg_MAE, MAE[0], batch_idx)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), torch.LongTensor(target).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) #already include softmax()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_idx , len(trainloader),
                100. * batch_idx / len(trainloader), loss.data[0]))
        if batch_idx % test_interval == 0 and batch_idx != 0:
            test( batch_idx, save=True)
            torch.save(model.state_dict(), '%sage_epoch_%d_%d.pth' % (outf, epoch, batch_idx))
    test(save=True)


def main():
    for epoch in range(1, epochs + 1):
        scheduler.step()
        train(epoch)
        torch.save(model.state_dict(), '%sage_epoch_%d.pth' % (outf, epoch))

if __name__ == "__main__":
    main()
    fsave.close()