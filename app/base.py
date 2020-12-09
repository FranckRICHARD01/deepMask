# -*- coding: utf-8 -*-
"""
Data Loading and Processing
requires:
-  ``scikit-image``: For image io and transforms
-  ``pandas``: For easier csv parsing
-  ``nibabel``: To read NIFTI files

"""

from __future__ import print_function, division
import os
import torch

from nibabel import load as load_nii
from skimage import transform as skt
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


######################################################################
# Dataset class
# -------------
#
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
#
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get :math:`i`\ th sample
#
# Let's create a dataset class for our face landmarks dataset. We will
# read the csv in ``__init__`` but leave the reading of images to
# ``__getitem__``. This is memory efficient because all the images are not
# stored in the memory at once but read as required.
#
# Sample of our dataset will be a dict
# ``{'image': image, 'landmarks': landmarks}``. Our datset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. We will see the usefulness of ``transform`` in the
# next section.
#

class InferMaskDataset(Dataset):
    def __init__(self, id, t1, flair, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.id = id
        self.t1 = t1
        self.t2 = flair
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return int(1)

    def __getitem__(self, idx):
        t1w_fname = os.path.join(self.root_dir, self.t1)
        t2w_fname = os.path.join(self.root_dir, self.t2)
        case_id = self.id

        t1w, t2w = load_nii(t1w_fname).get_data(), load_nii(t2w_fname).get_data()
        t1w = (t1w.astype(dtype=np.float32) - t1w[np.nonzero(t1w)].mean()) / t1w[np.nonzero(t1w)].std()
        t2w = (t2w.astype(dtype=np.float32) - t2w[np.nonzero(t2w)].mean()) / t2w[np.nonzero(t2w)].std()

        # header = load_nii(t1w_fname).header

        sample = {'t1w': t1w, 't2w': t2w, 'filename': t1w_fname, 'id': case_id}

        # affine = header.get_qform()
        # nii_out = nib.Nifti1Image(output_scan, affine, header)

        if self.transform:
            sample = self.transform(sample)

        return sample


######################################################################
# Transforms
# ----------
#
# One issue we can see from the above is that the samples are not of the
# same size. Most neural networks expect the images of a fixed size.
# Therefore, we will need to write some prepocessing code.
# Let's create three transforms:
#
# -  ``Rescale``: to scale the image
# -  ``ToTensor``: to convert the numpy images to torch images (we need to
#    swap axes).
#
# We will write them as callable classes instead of simple functions so
# that parameters of the transform need not be passed everytime it's
# called. For this, we just need to implement ``__call__`` method and
# if required, ``__init__`` method. We can then use a transform like this:
#
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
#
# Observe below how these transforms had to be applied both on the image and
# landmarks.
#


class inferResize(object):
    """Center crop the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        # t1w, t2w, labels = sample['t1w'], sample['t2w'], sample['labels']
        t1w, t2w, t1w_fname, case_id = sample['t1w'], sample['t2w'], sample['filename'], sample['id']
        # t1w = skt.resize(t1w, self.output_size, anti_aliasing=True)
        t1w = skt.resize(t1w, self.output_size, mode='constant', preserve_range=1)
        t2w = skt.resize(t2w, self.output_size, mode='constant', preserve_range=1)

        # return {'t1w': t1w, 't2w': t2w, 'labels': labels}
        return {'t1w': t1w, 't2w': t2w, 'filename': t1w_fname, 'id': case_id}


class ToTensorInfer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        t1w, t2w, t1w_fname, case_id = sample['t1w'], sample['t2w'], sample['filename'], sample['id']

        image = np.stack((t1w, t2w), axis=0)

        return {'image': torch.from_numpy(image),
                'filename': t1w_fname,
                'id': case_id}