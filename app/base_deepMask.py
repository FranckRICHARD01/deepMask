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
import pandas as pd
from nibabel import load as load_nii
from skimage import transform as skt
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.utils import class_weight

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


def compute_weights(labels, binary=False):
	if binary:
		labels = (labels > 0).astype(np.int_)

	weights = class_weight.compute_class_weight('balanced', np.unique(labels.flatten()), labels.flatten())
	# if labels.ndim > 3:
	# 	weights  = np.array(weights.mean(axis=0))
	return weights



def dice_subfields(image, label, label_num, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """

    image = np.where(image == label_num, 1, 0)

    image = np.asarray(image).astype(np.bool)
    label = np.asarray(label).astype(np.bool)

    if image.shape != label.shape:
        raise ValueError("Shape mismatch: image {0} and label {1} must have the same shape.".format(image.shape, label.shape))

    im_sum = image.sum() + label.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(image, label)

    return 2. * intersection.sum() / im_sum



def dice_gross(image, label, empty_score=1.0):

    image = (image > 0).astype(np.int_)
    label = (label > 0).astype(np.int_)

    image = np.asarray(image).astype(np.bool)
    label = np.asarray(label).astype(np.bool)

    if image.shape != label.shape:
        raise ValueError("Shape mismatch: image {0} and label {1} must have the same shape.".format(image.shape, label.shape))

    im_sum = image.sum() + label.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(image, label)

    return 2. * intersection.sum() / im_sum


def get_nii_hdr_affine(t1w_fname):
    nifti = load_nii(t1w_fname)
    shape = nifti.get_data().shape
    header = load_nii(t1w_fname).header
    affine = header.get_qform()
    return nifti, header, affine, shape


import matplotlib.cm as cm
from scipy import ndimage
def plotslice(i, image, label, ax, fig, x, y, z, alfa=0.5, binarize=False, mask=False):

    if binarize:
        label = (label > 0).astype(np.int_)

    if mask:
        label = (label > 0).astype(np.int_)
        t1w_bin = (image > 0).astype(np.int_)
        mask = t1w_bin + label
        label = t1w_bin

    ax = plt.subplot(1, 3, 1)
#     plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(ndimage.rotate(image[x,:,:],90), cmap=cm.gray)
    plt.imshow(ndimage.rotate(label[x,:,:],90), cmap=cm.viridis, alpha=alfa)

    ax = plt.subplot(1, 3, 2)
#     plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(ndimage.rotate(image[:,y,:],90), cmap=cm.gray)
    plt.imshow(ndimage.rotate(label[:,y,:],90), cmap=cm.viridis, alpha=alfa)

    ax = plt.subplot(1, 3, 3)
#     plt.tight_layout()
    ax.set_title('Sample #{}'.format(i), )
    ax.axis('off')
    plt.imshow(ndimage.rotate(image[:,:,z],90), cmap=cm.gray)
    plt.imshow(ndimage.rotate(label[:,:,z],90), cmap=cm.viridis, alpha=alfa)

######################################################################
# Iterating through the dataset
# -----------------------------
#
# Let's put this all together to create a dataset with composed
# transforms.
# To summarize, every time this dataset is sampled:
#
# -  An image is read from the file on the fly
# -  Transforms are applied on the read image
# -  Since one of the transforms is random, data is augmentated on
#    sampling
#
# We can iterate over the created dataset with a ``for i in range``
# loop as before.
#

# transformed_dataset = HCPDataset(csv_file='HCP_371.csv',
#                                  root_dir='/host/hamlet/local_raid/data/ravnoorX/deepHC/data/',
#                                  transform=transforms.Compose([CenterCrop(256),ToTensor()])
#                                  )
#
# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]
#
#     print(i, sample['image'].size(), sample['landmarks'].size())
#
#     if i == 3:
#         break
