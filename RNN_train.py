import os
import sys
import cv2
#import openslide
from PIL import Image
import numpy as np
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 RNN aggregator training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--s', default=10, type=int, help='how many top k tiles to consider (default: 10)')
parser.add_argument('--ndims', default=128, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--model', type=str, help='path to trained model checkpoint')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--shuffle', default=False, action='store_true', help='to shuffle order of sequence')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()

    #load libraries

    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

   
    train_dset = rnndata(args.train_lib, args.s, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_dset = rnndata(args.val_lib,  args.s, trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    #make model
    embedder = ResNetEncoder(args.model)
    for param in embedder.parameters():
        param.requires_grad = False
    embedder = embedder.cuda()
    embedder.eval()

    rnn = rnn_single(args.ndims)
    rnn = rnn.cuda()

    #optimization
    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    cudnn.benchmark = True

    fconv = open(os.path.join(args.output, 'convergence_{}.csv'.format(str(args.s))), 'w')
    fconv.write('epoch,train.loss,train.fpr,train.fnr,val.loss,val.fpr,val.fnr\n')
    fconv.close()

    #
    last_epoch_since_improvement = 0
    for epoch in range(args.nepochs):
        if last_epoch_since_improvement <= 20:
            train_loss, train_fpr, train_fnr = train_single(epoch, embedder, rnn, train_loader, criterion, optimizer)
            val_loss, val_fpr, val_fnr = test_single(epoch, embedder, rnn, val_loader, criterion)

            fconv = open(os.path.join(args.output,'convergence_{}.csv'.format(str(args.s))), 'a')
            fconv.write('{},{},{},{},{},{},{}\n'.format(epoch+1, train_loss, train_fpr, train_fnr, val_loss, val_fpr, val_fnr))
            fconv.close()

            val_err = (val_fpr + val_fnr)/2
            if 1-val_err >= best_acc:
                if 1-val_err - best_acc > .05:
                    last_epoch_since_improvement = 0
                best_acc = 1-val_err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': rnn.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'rnn_checkpoint_best_{}_best_acc_{}.pth'.format(str(args.s), str(round(best_acc, 2)))))
            else:
                print('last_epoch_since_improvement  ', last_epoch_since_improvement)
                last_epoch_since_improvement += 1
        else:
            break

 

def train_single(epoch, embedder, rnn, loader, criterion, optimizer):
    rnn.train()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    for i,(inputs,target) in enumerate(loader):
        print('Training - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1, args.nepochs, i+1, len(loader)))

        batch_size = inputs[0].size(0)
        rnn.zero_grad()

        state = rnn.init_hidden(batch_size).cuda()
        print('state ', state.size())
        for s in range(len(inputs)):
            input = inputs[s].cuda()
            _, input = embedder(input)
            output, state = rnn(input, state)

        target = target.cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*target.size(0)
        fps, fns = errors(output.detach(), target.cpu())
        running_fps += fps
        running_fns += fns

    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    print('Training - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def test_single(epoch, embedder, rnn, loader, criterion):
    rnn.eval()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            print('Validating - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1,args.nepochs,i+1,len(loader)))

            batch_size = inputs[0].size(0)

            state = rnn.init_hidden(batch_size).cuda()
            for s in range(len(inputs)):
                input = inputs[s].cuda()
                _, input = embedder(input)
                output, state = rnn(input, state)

            target = target.cuda()
            loss = criterion(output,target)

            running_loss += loss.item()*target.size(0)
            fps, fns = errors(output.detach(), target.cpu())
            running_fps += fps
            running_fns += fns

    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    print('Validating - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def errors(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    real = target.numpy()
    neq = pred!=real
    fps = float(np.logical_and(pred==1,neq).sum())
    fns = float(np.logical_and(pred==0,neq).sum())
    return fps,fns

class ResNetEncoder(nn.Module):

    def __init__(self, path):
        super(ResNetEncoder, self).__init__()

        temp = models.resnet34()
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        ch = torch.load(path)
        temp.load_state_dict(ch['state_dict'])
        self.features = nn.Sequential(*list(temp.children())[:-1])
        self.fc = temp.fc

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.fc(x), x

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        print('Emilie ', 'encoder_dim ',encoder_dim, 
             'decoder_dim ', decoder_dim,
              'attention_dim  ', attention_dim)
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
# class rnn_single(nn.Module):

#     def __init__(self, ndims):
#         super(rnn_single, self).__init__()
#         self.ndims = ndims

#         self.fc1 = nn.Linear(512, ndims)
#         self.fc2 = nn.Linear(ndims, ndims)

#         self.fc3 = nn.Linear(ndims, 2)

#         self.activation = nn.ReLU()

#     def forward(self, input, state):
#         input = self.fc1(input)
#         state = self.fc2(state)
#         state = self.activation(state+input)
#         output = self.fc3(state)
#         return output, state

#     def init_hidden(self, batch_size):
#         return torch.zeros(batch_size, self.ndims)

class rnn_single(nn.Module):

    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims # 128
        self.decoder_dims = 128
        self.attention_dims = 512
        self.attention = Attention(512, self.ndims, self.attention_dims)
        self.fc1 = nn.Linear(512, ndims)
        self.Testfc1 = nn.Linear(512, ndims)
        self.fc2 = nn.Linear(ndims, ndims)

        self.fc3 = nn.Linear(ndims, 2)

        self.activation = nn.ReLU()

    def forward(self, input, state):
        attention_weighted_encoding, alpha = self.attention(input,
                                                               state)
        input_weighted = torch.mul( input, attention_weighted_encoding)
        test = self.Testfc1(input_weighted)
        input = self.fc1(input_weighted)
        state = self.fc2(state)
        state = self.activation(state+input)
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        self.h = torch.zeros(batch_size, self.ndims)
        return torch.zeros(batch_size, self.ndims)

class rnndata(data.Dataset):

    def __init__(self, libraryfile, s, transform=None, shuffle=None):
        with open(libraryfile) as json_file:
            lib = json.load(json_file)
        slides = lib['Slides']
#
        tiles_full = []
        print(len(lib['Tiles']))
        for i,g in enumerate(lib['Tiles']):
            #print('g' , g)
            tiles_full.extend(g)

        print('Number of tiles: {}'.format(len(tiles_full)))
        print('Length ', len(tiles_full))
        self.slidenames = lib['Slides']
        self.targets = lib['Targets']
        self.tiles = lib['Tiles']
        self.tiles_full = tiles_full
        self.transform = transform
        self.s = s
        self.shuffle = shuffle

    def __getitem__(self,index):
        tiles_path= random.sample(self.tiles[index],len(self.tiles[index]))
        out = []
        for i in range(self.s):
            #print('tiles_path[i]  ', tiles_path[i])
            img = cv2.imread(tiles_path[i])
            #print('v \n')
            img  = cv2.resize(img, (224,224), interpolation = cv2.INTER_LINEAR )
            if self.transform is not None:
                img = self.transform(img)
            out.append(img)
        return out, self.targets[index]

    def __len__(self):

        return len(self.targets)

if __name__ == '__main__':
    main()
