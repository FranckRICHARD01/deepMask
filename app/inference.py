#!/usr/bin/env python3
"""
usage:
inference.py ${id} ${id}_t1.nii.gz ${id}_flair.nii.gz ${IN_DIR} ${OUT_DIR}
"""

import os, sys, time

import torch
import torch.nn as nn
from mo_dots import Data
import multiprocessing

from utils.data import *
from deepmask import *
import vnet

# configuration
args = Data()
args.cpus = multiprocessing.cpu_count()

args.outdir = '/tmp'

args.model = 'vnet.masker.20180316_0441'
# training based on manually corrected masks from
# 153 patients with cortical malformations
args.inference = '/app/weights/vnet_masker_model_best.pth.tar' 

# resize all input images to this resolution  matching training data
args.resize = (160,160,160)

args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("build vnet, using GPU")
else:
    print("build vnet, using CPU")


def build_model(args):
    # do not change these variable for inference
    model = vnet.VNet(n_filters=6, outChans=2, elu=True, nll=True)
    model = nn.parallel.DataParallel(model, device_ids=device)

    if os.path.isfile(args.inference):
        print("=> loading checkpoint '{}'".format(args.inference))
        if args.cuda:
            checkpoint = torch.load(args.inference)
        else:
            checkpoint = torch.load(args.inference, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    else:
        sys.exit("=> no checkpoint found at '{}'".format(args.inference))

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.cuda:
        print('moving the model to GPU')
        model = model.cuda()

    return model