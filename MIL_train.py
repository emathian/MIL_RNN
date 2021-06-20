import sys
import os
import json
import numpy as np
import argparse
import random
#import openslide
import cv2
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import StepLR
from PIL import Image


parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size_train', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--batch_size_val', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
parser.add_argument('--previous_checkpoint', default=None, type=str, help='Path to the previous checkopoint if the training has been interupted')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()
    #cudnn
    
    if args.previous_checkpoint is None:
        model = ConvNet()
        #model.classifier[6] = nn.Linear(4096,2)
        #model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        model = models.alexnet(args.previous_checkpoint)
        model.classifier[6] = nn.Linear(4096,2)
        #model.fc = nn.Linear(model.fc.in_features, 2)
    model.cuda()


    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    lr_ = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr_, weight_decay=1e-4)
    scheduler =  StepLR(optimizer, step_size=10, gamma=0.1)

    cudnn.benchmark = True

    #normalization
    #normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    color = transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.3, hue=0.02)
    ##trans = transforms.Compose([transforms.ToTensor(), color])
    trans = transforms.Compose([ color, transforms.ToTensor()])



    #load data
    train_dset = MILdataset(args.train_lib, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size_train, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size_val, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    #open output file
    fconv = open(os.path.join(args.output,'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #loop throuh epochs
    for epoch in range(args.nepochs):
        if epoch > 3 and epoch <= 6:
            lr_ = 1e-4
            optimizer.param_groups[0]['lr'] = lr_
        elif epoch > 6 and epoch <= 18:
            lr_ = 1e-4
            optimizer.param_groups[0]['lr'] = lr_
        else:
            scheduler.step()
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        train_dset.setmode(1)
        probs = inference(epoch, train_loader, model, 'train')
        maxs = group_max(np.array(train_dset.slideIDX), probs, len(train_dset.targets))
        pred = [1 if x >= 0.5 else 0 for x in maxs]
        err,fpr,fnr = calc_err(pred, train_dset.targets)
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},Training_error,{}\n'.format(epoch+1, err))
        fconv.write('{},Training_fpr,{}\n'.format(epoch+1, fpr))
        fconv.write('{},Training_fnr,{}\n'.format(epoch+1, fnr))
        fconv.write('{},Training_fnr,{}\n'.format(epoch+1, str(optimizer.param_groups[0]['lr'])))
        fconv.close()
        
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        #Validation
        if args.val_lib : # and (epoch+1) % args.test_every == 0
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model, 'eval')
            maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err,fpr,fnr = calc_err(pred, val_dset.targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #Save best model
            err = (fpr+fnr)/2.
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'checkpoint_best_{}.pth'.format(str( epoch+1))))

def inference(run, loader, model, eval_train):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            if eval_train == 'train':
                probs[i*args.batch_size_train:i*args.batch_size_train+input.size(0)] = output.detach()[:,1].clone()
            else:
                probs[i*args.batch_size_val:i*args.batch_size_val+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)

    return running_loss/len(loader.dataset)

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):

    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out

class ConvNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.Conv2d(64, 192, kernel_size=5, padding=2),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192, momentum=0.01),
#             nn.Conv2d(192, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        with open(libraryfile) as json_file:
            lib = json.load(json_file)
        slides = lib['Slides']
#         for i,name in enumerate(lib['slides']):
#             sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
#             sys.stdout.flush()
#             slides.append(openslide.OpenSlide(name))
        #Flatten grid
        tiles_full = []
        slideIDX = []
        print(len(lib['Tiles']))
        for i,g in enumerate(lib['Tiles']):
            #print('g' , g)
            tiles_full.extend(g)
            slideIDX.extend([i]*len(g))

        print('Number of tiles: {}'.format(len(tiles_full)))
        print('Length ', len(tiles_full), len(slideIDX))
        self.slidenames = lib['Slides']
        self.targets = lib['Targets']
        self.tiles = lib['Tiles']
        self.tiles_full = tiles_full
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None

    def setmode(self,mode):
        print('mode ', mode)
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.tiles_full[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            tiles_path = self.tiles_full[index]
            ##img = cv2.imread(tiles_path)
            img = Image.open(tiles_path)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            tiles_path = self.tiles_full[index]
            try:
                img = Image.open(tiles_path)
                #img = cv2.imread(tiles_path)
            except:
                print('ERROR    ', tiles_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.tiles_full)
        elif self.mode == 2:
            return len(self.t_data)

if __name__ == '__main__':
    main()
