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

    # #############################################################
    # 1) Initialize the dataset, model, optimizer and loss as usual.
    # Initialize a fake dataset


    global args, best_acc
    args = parser.parse_args()

    #cudnn
    if args.previous_checkpoint is None:
        model = models.resnet34(True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        model = models.resnet34(args.previous_checkpoint)
        model.fc = nn.Linear(model.fc.in_features, 2)
    model.cuda()

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = MILdataset(args.train_lib, trans)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, trans)
    # #############################################################
    # 2) Set parameters for the adaptive batch size
    adapt = True  # while this is true, the algorithm will perform batch adaptation
    gpu_batch_size = 128  # initial gpu batch_size, it can be super small
    train_batch_size = 512  # the train batch size of desire
    continue_training = True
    # Modified training loop to allow for adaptive batch size
    while continue_training:

        # #############################################################
        # 3) Initialize dataloader and batch spoofing parameter
        # Dataloader has to be reinicialized for each new batch size.
        train_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=int(gpu_batch_size), shuffle=False,
            num_workers=args.workers, pin_memory=False)


        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=int(gpu_batch_size), shuffle=False,
            num_workers=args.workers, pin_memory=False)
        print('\n\n\n ***********************************************************************\n int(gpu_batch_size)   ',
        int(gpu_batch_size), '\n\n\n ***********************************************************************')


        # Number of repetitions for batch spoofing
        repeat = max(1, int(train_batch_size / gpu_batch_size))

        try:  # This will make sure that training is not halted when the batch size is too large

            # #############################################################
            # 4) Epoch loop with batch spoofing
            optimizer.zero_grad()  # done before training because of batch spoofing.

            # Emi

            #open output file
            fconv = open(os.path.join(args.output,'convergence.csv'), 'w')
            fconv.write('epoch,metric,value\n')
            fconv.close()

            #loop throuh epochs
            for epoch in range(args.nepochs):

                train_dset.setmode(1)
                probs = inference(epoch, train_loader, model, 'train')
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
                #         print('args.val_lib  ', args.val_lib)
                #         if args.val_lib :# and (epoch+1) % args.test_every == 0
                val_dset.setmode(1)
                probs = inference(epoch, val_loader, model, 'val')
                maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
                pred = [1 if x >= 0.5 else 0 for x in maxs]
                err,fpr,fnr = calc_err(pred, val_dset.targets)
                print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
                fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
                fconv.write('{},error,{}\n'.format(epoch+1, err))
                fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
                fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
                fconv.close()


                flog = open(os.path.join(args.output,'log_BatchSize.csv'), 'a')
                flog.write('Compatble Batch Size = {} \n'.format(str(gpu_batch_size)))
                flog.close()
                # #Save best model
                # err = (fpr+fnr)/2.
                # print('best_acc  ', best_acc, ' 1-err  ',  1-err)
                # if 1-err >= best_acc:
                #     print('1-err  ', 1-err, 'best_acc  ', best_acc)
                #     best_acc = 1-err
                #     obj = {
                #         'epoch': epoch+1,
                #         'state_dict': model.state_dict(),
                #         'best_acc': best_acc,
                #         'optimizer' : optimizer.state_dict()
                #     }
                #     print('SaVE')
                #     torch.save(obj, os.path.join(args.output,'checkpoint_best.pth'))

                # #############################################################
                # 5) Adapt batch size while no RuntimeError is rased.
                # Increase batch size and get out of the loop
                if adapt:
                    gpu_batch_size *= 2
                    break

                if epoch > 1:
                    continue_training = False



                # #############################################################
                # 5) Adapt batch size while no RuntimeError is rased.
                # Increase batch size and get out of the loop
                if adapt:
                    gpu_batch_size *= 2
                    break

                # Stopping criteria for training
                if i > 100:
                    continue_training = False

        # #############################################################
        # 6) After the largest batch size is found, the training progresses with the fixed batch size.
        # CUDA out of memory is a RuntimeError, the moment we will get to it when our batch size is too large.
        except RuntimeError as run_error:

            flog = open(os.path.join(args.output,'log_BatchSize.csv'), 'a')
            flog.write('Max Batch Size = {} \n'.format(str(gpu_batch_size)))
            flog.close()


            gpu_batch_size /= 2  # resize the batch size for the biggest that works in memory
            adapt = False  # turn off the batch adaptation

            # Number of repetitions for batch spoofing
            repeat = max(1, int(train_batch_size / gpu_batch_size))

            # Manual check if the RuntimeError was caused by the CUDA or something else.
            print(f"---\nRuntimeError: \n{run_error}\n---\n Is it a cuda error?")








def inference(run, loader, model, eval_train):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
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
    print('data  ', data)
    print('groups  ', groups)
    print(type(data), type(groups))
    print(data.shape)
    print(groups.shape)
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
