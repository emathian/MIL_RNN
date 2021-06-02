import sys
import os
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
import json
parser = argparse.ArgumentParser(description='')
parser.add_argument('--lib', type=str, default='filelist', help='path to data file')
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--model', type=str, default='', help='path to pretrained model')
parser.add_argument('--batch_size', type=int, default=100, help='how many images to sample per slide (default: 100)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')

def main():
    global args
    args = parser.parse_args()

    #load model
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ch = torch.load(args.model)
    model.load_state_dict(ch['state_dict'])
    model = model.cuda()
    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(),normalize])

    #load data
    dset = MILdataset(args.lib, trans)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    dset.setmode(1)
    probs = inference(loader, model)
    print('probs ', len(probs))
    print('np.array(dset.slideIDX)  ', np.array(dset.slideIDX).shape)
    print('len(dset.targets)  ',len(dset.targets))
    maxs = group_max(np.array(dset.slideIDX), probs, len(dset.targets))


    fpt = open(os.path.join(args.output, 'probability.csv'), 'w')
    fpt.write('Tiles,target,probability\n')
    for slide, tiles_l, targets in  zip(dset.slidenames,dset.tiles, dset.targets):
        for name, prob in zip(tiles_l, probs):
            fpt.write('{},{},{},{}\n'.format(slide,name, targets, prob))
    fpt.close()


    fp = open(os.path.join(args.output, 'predictions.csv'), 'w')
    fp.write('file,target,prediction,probability\n')
    for name, target, prob in zip(dset.slidenames, dset.targets, maxs):
        fp.write('{},{},{},{}\n'.format(name, target, int(prob>=0.5), prob))
    fp.close()

def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Batch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

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
    return list(out)


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
        print('len(lib[Tiles])  ',len(lib['Tiles']))
        print('lib[Slides]  ', len(lib['Slides']))
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
#         self.mult = lib['mult']
#         self.size = int(np.round(224*lib['mult']))
#         self.level = lib['level']
    def setmode(self,mode):
        print('mode ', mode)
        self.mode = mode
    def maketraindata(self, idxs):
        print('idxs  ',idxs)
        self.t_data = [(self.slideIDX[x],self.tiles_full[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            tiles_path = self.tiles_full[index]
            img = cv2.imread(tiles_path)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            tiles_path = self.tiles_full[index]
            img = cv2.imread(tiles_path)
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
