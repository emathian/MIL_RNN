
import os
import datetime
import copy
import re
import uuid
import warnings
import time
import inspect
import json
import numpy as np
import pandas as pd
from functools import partial, reduce
from random import shuffle
import random
import torch
from torch import nn, optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision.models import resnet
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
import random 
import multiprocessing as mp
import os
global path_to_digit_folder
path_to_digit_folder  = '/home/mathiane/Downloads/trainingSet/trainingSet'
def CreateAbag(path_to_digit_folder):
    L_tiles = []
    pos_neg_p = random.random()
    positive_bag = False
    Target = 0 
    possible_element = [0,1,2,3,4,5,6,7,8]
    if pos_neg_p < 0.5:
        positive_bag = True
        Target = 1 
    Nb_element = 150#random.randint(100,700)
    print('positive_bag  ', positive_bag)
    if positive_bag ==  False:
        while len(L_tiles) < Nb_element:
            random.shuffle(possible_element)
            folder_number = str(possible_element[0])
            list_picts = os.listdir(os.path.join(path_to_digit_folder,
                                                 folder_number))
            random.shuffle(list_picts)
            L_tiles.append(os.path.join(path_to_digit_folder,
                                       folder_number , list_picts[0]))
    else:
        p80 = int(0.8 * Nb_element)
        folder_number = str(9)
        list_picts = os.listdir(os.path.join(path_to_digit_folder,
                                                 folder_number))
        L_tiles += [os.path.join(path_to_digit_folder,
                                                 folder_number, i)for i in list_picts[:p80]]
        random.shuffle(list_picts)
        print('Nb_element  ', Nb_element,'len list bf while  ', len(L_tiles))
        while len(L_tiles) < Nb_element:
            random.shuffle(possible_element)
            folder_number = str(possible_element[0])
            list_picts = os.listdir(os.path.join(path_to_digit_folder,
                                                 folder_number))
            random.shuffle(list_picts)
            
            L_tiles.append(os.path.join(path_to_digit_folder,
                                       folder_number , list_picts[0]))
        random.shuffle(L_tiles)
    return L_tiles, Target

def create_set_bags(nb_bags = 20):
    Tiles = []
    Slides = []
    Targets = []
    for i in range(nb_bags):
        Bags_name = 'Bag_{}'.format(str(i))
        L_tiles, Target = CreateAbag(path_to_digit_folder)
        Tiles.append(L_tiles)
        Targets.append(Target)
        Slides.append(Bags_name)
    return Slides, Tiles, Targets



a_pool = mp.Pool()
s1, s2, s3, s4 = a_pool.map(create_set_bags,[10000] * 4) 
Slides = s1[0] + s2[0] + s3[0] + s4[0]
Tiles = s1[1] + s2[1] + s3[1] + s4[1]
Targets = s1[2] + s2[2] + s3[2] + s4[2]

DictTarinMnist = {'Slides': Slides,
             'Tiles': Tiles,
            'Targets' : Targets }

with open('TrainMNIST.json', 'w') as fp:
    json.dump(DictTarinMnist, fp)