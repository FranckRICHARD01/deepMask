#!/usr/bin/env python3

# source activate torch3
# deepMask-run_postCIVET.py ${id} ${id}_t1_final.nii.gz ${id}_flair_final.nii.gz /host/silius/local_raid/ravnoor/01_Projects/06_DeepLesion_LoSo/data/final /host/silius/local_raid/ravnoor/01_Projects/06_DeepLesion_LoSo/data/deepMasks

import os, math, sys, time, fileinput, re

import subprocess
from subprocess import Popen, PIPE

from base_deepMask import *

import torch

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import shutil, setproctitle

import vnet
# import make_graph
from functools import reduce
import operator
from itertools import starmap

import nibabel as nib
from nibabel.processing import resample_to_output as resample
from mo_dots import wrap, Data

import pandas as pd
from skimage import transform as skt
from sklearn import metrics


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def inference(args, loader, model, t2w_fname, nifti=False):
    src = args.inference
    dst = args.outdir+'/'+args.model

    model.eval()
    # nvols = reduce(operator.mul, target_split, 1)
    # assume single GPU / batch size 1
    for sample in loader:
        start_time = time.time()

        data, case_id, t1w_fname = sample['image'], sample['id'], sample['filename']

        _, header, affine, out_shape = get_nii_hdr_affine(t1w_fname[0]) # load original input with header and affine

        shape = data.size()
        # convert names to batch tensor
        if args.cuda:
            data.pin_memory()
            data = data.cuda().float()
        else:
            data = data.float()
        data = Variable(data, volatile=True)
        output = model(data)
        # _, output = output.max(1)
        output = torch.argmax(output, dim=1)
        output = output.view(shape[2:])
        output = output.cpu()
        output = output.data.numpy()

        print("save {}".format(case_id[0]))
        if not os.path.exists(os.path.join(dst)):
            os.makedirs(os.path.join(dst), exist_ok=True)

        if nifti:
            affine = header.get_qform()
            output = skt.resize(output, output_shape=out_shape, order=1, mode='wrap', preserve_range=1, anti_aliasing=True)
            # output = np.where(output>0.5, 1, 0).astype(np.int_)
            nii_out = nib.Nifti1Image(output, affine, header)
            # nii_out = resample(in_img=nii_out, out_class=nii_self_template)
            nii_out.to_filename(os.path.join(dst, case_id[0]+"_vnet_maskpred.nii.gz"))

        elapsed_time = time.time() - start_time
        print("=> inference time: {} seconds".format(round(elapsed_time,2)))
        print("=*80")

    config='/host/hamlet/local_raid/data/ravnoor/01_Projects/12_deepMask/src/noel_CIVET_masker/config_densecrf.txt'
    # t2w_fname = re.sub('T1', 'FLAIR', t1w_fname[0])
    print(t1w_fname[0], t2w_fname[0])
    start_time = time.time()
    denseCRF(case_id[0], t1w_fname[0], t2w_fname, out_shape, config, dst, dst, os.path.join(dst, case_id[0]+"_vnet_maskpred.nii.gz"))
    elapsed_time = time.time() - start_time
    print("=*80")
    print("=> dense 3D-CRF inference time: {} seconds".format(round(elapsed_time,2)))
    print("=*80")


def denseCRF(id, t1, t2, input_shape, config, in_dir, out_dir, pred_labels):
    X, Y, Z = input_shape
    config_tmp = "/tmp/"+id+"_config_densecrf.txt"
    print(config_tmp)
    subprocess.call(["cp", "-f", config, config_tmp])
    find_str = ["<ID_PLACEHOLDER>", "<T1_FILE_PLACEHOLDER>", "<FLAIR_FILE_PLACEHOLDER>", "<OUTDIR_PLACEHOLDER>", "<PRED_LABELS_PLACEHOLDER>", "<X_PLACEHOLDER>", "<Y_PLACEHOLDER>", "<Z_PLACEHOLDER>"]
    replace_str = [str(id), str(t1), str(t2), str(out_dir), str(pred_labels), str(X), str(Y), str(Z)]
    # config_tmp_replicate = [x for x in [config_tmp] for _ in range(len(find_str))]
    # [print(a,b,c) for a,b,c in zip(config_tmp_replicate, find_str, replace_str)]
    # starmap(find_replace_re, zip(config_tmp_replicate, find_str, replace_str))
    for fs, rs in zip(find_str, replace_str):
        find_replace_re(config_tmp, fs, rs)
    subprocess.call(["/host/hamlet/local_raid/data/ravnoor/01_Projects/12_deepMask/src/densecrf/dense3dCrf/dense3DCrfInferenceOnNiis", "-c", config_tmp])


def find_replace_re(config_tmp, find_str, replace_str):
    with fileinput.FileInput(config_tmp, inplace=True, backup='.bak') as file:
        for line in file:
            print(re.sub(find_str, str(replace_str), line.rstrip(), flags=re.MULTILINE), end='\n')

args=Data()
args.batchSz=3
args.ngpu=1

args.evaluate=''

args.modeldir = '/host/hamlet/local_raid/data/ravnoor/01_Projects/12_deepMask/src/work/'
args.outdir = sys.argv[5]

# args.save='vnet.masker.20180309_1316' # training, N=133
args.model='vnet.masker.20180316_0441' # training, N=153
args.inference=args.modeldir+args.model+'/vnet_masker_model_best.pth.tar'

args.seed=1
args.opt='adam'
args.nEpochs=100

args.cuda = torch.cuda.is_available()
print(args.cuda)
# args.cuda = False

setproctitle.setproctitle(args.model+'_'+sys.argv[1])

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.cuda:
    print("build vnet, using GPU")
else:
    print("build vnet, using CPU")

start_time = time.time()
model = vnet.VNet(n_filters=6, outChans=2, elu=True, nll=True)
batch_size = args.ngpu*args.batchSz
gpu_ids = range(args.ngpu)

if args.cuda:
    model = nn.parallel.DataParallel(model, device_ids=gpu_ids)

if os.path.isfile(args.inference):
    print("=> loading checkpoint '{}'".format(args.inference))
    checkpoint = torch.load(args.inference)
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

resize = (160,160,160)

inferenceSet = InferMaskDataset(id=sys.argv[1],
                                t1=sys.argv[2],
                               flair=sys.argv[3],
                               root_dir=sys.argv[4],
                               transform=transforms.Compose([ inferResize(resize), ToTensorInfer() ])
                               )

kwargs = {'num_workers': 1} if args.cuda else {}
# src = args.inference
# dst = args.save
# inference_batch_size = args.ngpu
loader = DataLoader(inferenceSet, batch_size=1, shuffle=False, **kwargs) # do not change batch_size=1

inference(args, loader, model, t2w_fname=sys.argv[4]+sys.argv[3], nifti=True) # corresponds to InferMaskDataset
