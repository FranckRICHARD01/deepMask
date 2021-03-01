#!/usr/bin/env python3

import os
import torch
from mo_dots import Data
import multiprocessing
from utils.data import *
from deepmask import *
import vnet
from utils.image_processing import noelImageProcessor

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
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("build vnet, using GPU")
else:
    print("build vnet, using CPU")

model = vnet.build_model(args)

template = os.path.join('./template', 'mni_icbm152_t1_tal_nlin_sym_09a.nii.gz')

noelImageProcessor(id=str(case_id), t1=t1, t2=t2, output_dir=args.outdir, template=template, usen3=False, args=args, model=model).preprocessor()