
from sklearn.utils import class_weight
import matplotlib.cm as cm
from scipy import ndimage

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
