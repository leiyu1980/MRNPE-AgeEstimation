import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as scheduler

batch_size = 64
test_batch_size = 64
epochs = 12
lr = 0.001
momentum = 0.9
weight_decay = 0.0005
stepsize = 4
log_interval = 100
gamma = 0.1
test_interval = 1000
outf = "./snapshot"
root = "/fast/age/"
train_label = "/fast/age/224dataset/s1_train/Face_train_s.txt"
test_label = "/fast/age/224dataset/s1_val/face_val.txt"

import os
import torch
import torch.utils.data as data
from PIL import Image

mytransform = transforms.Compose([
    transforms.ToTensor()
    ]
)

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 101),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

train_set  = myImageFloder(root = root, label = train_label, transform = mytransform)
trainloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=False, num_workers=2)
test_set  = myImageFloder(root = root, label = test_label , transform = mytransform)
testloader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size, shuffle=True, num_workers=2)
model = AlexNet()
model.apply(weights_init)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = scheduler.StepLR(optimizer, step_size=stepsize, gamma = gamma)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    val_len = 10000
    for data, target in testloader:
        data, target = data.cuda(), torch.LongTensor(target).cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        #test_loss += torch.nn.MultiLabelSoftMarginLoss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cuda().sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, val_len,
        100. * correct / val_len))

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), torch.LongTensor(target).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) #already include softmax()
        #loss = torch.nn.MultiLabelSoftMarginLoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(trainloader),
                100. * batch_idx / len(trainloader), loss.data[0]))
        if batch_idx % test_interval == 0:
            test()

def main():
    for epoch in range(1, epochs + 1):
        scheduler.step()
        train(epoch)
        torch.save(model.state_dict(), '%sage_epoch_%d.pth' % (outf, epoch))

if __name__ == "__main__":
    main()
