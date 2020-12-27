#!/usr/bin/env python3
"""
usage:
inference.py ${id} ${id}_t1.nii.gz ${id}_flair.nii.gz ${IN_DIR} ${OUT_DIR}
"""

import os, math, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from mo_dots import Data
import setproctitle

import multiprocessing

from base import *
from utils import *
import vnet

# configuration
args = Data()
args.batchSz = 3
args.ngpu = 1
args.cpus = multiprocessing.cpu_count()

args.evaluate = ''

args.outdir = '/tmp'

# args.save='vnet.masker.20180309_1316' # training, N=133
args.model = 'vnet.masker.20180316_0441'
args.inference = '/app/models/vnet_masker_model_best.pth.tar' # vnet.masker.20180316_0441' # training, N=153

args.seed = 1
args.opt = 'adam'
args.nEpochs = 100

# resize all input images to this resolution  matching training data
resize = (160,160,160)

args.cuda = torch.cuda.is_available()

# set process title for htop/nvidia-smi/ytop monitoring
setproctitle.setproctitle(args.model + '_' + str(sys.argv[1]))

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("build vnet, using GPU")
else:
    print("build vnet, using CPU")

start_time = time.time()
model = vnet.VNet(n_filters=6, outChans=2, elu=True, nll=True)
batch_size = args.ngpu * args.batchSz
gpu_ids = range(args.ngpu)

model = nn.parallel.DataParallel(model, device_ids=device)

if os.path.isfile(args.inference):
    print("=> loading checkpoint '{}'".format(args.inference))
    if args.cuda:
        checkpoint = torch.load(args.inference)
    else:
        checkpoint = torch.load(args.inference, map_location=torch.device('cpu'))
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    elapsed_time = time.time() - start_time
    print("=> loaded checkpoint '{}' (epoch {}) in {} seconds"
          .format(args.model, checkpoint['epoch'], round(elapsed_time,2)))
else:
    sys.exit("=> no checkpoint found at '{}'".format(args.inference))


print('  + Number of params: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

if args.cuda:
    print('moving the model to GPU')
    model = model.cuda()

inferenceSet = InferMaskDataset(
                                id = sys.argv[1],
                                t1 = sys.argv[2], flair = sys.argv[3],
                                root_dir = sys.argv[4],
                                transform = transforms.Compose([ inferResize(resize), ToTensorInfer() ])
                               )

kwargs = {'num_workers': 1} if args.cuda else {'num_workers': args.cpus // 2}

loader = DataLoader(inferenceSet, batch_size=1, shuffle=False, **kwargs) # do not change batch_size=1

inference(args, loader, model, t2w_fname=os.path.join(sys.argv[4], sys.argv[3]), nifti=True) # corresponds to InferMaskDataset
