import re, fileinput
from sklearn.utils import class_weight
import matplotlib.cm as cm
from scipy import ndimage


def inference(args, loader, model, t2w_fname, nifti=False):
    src = args.inference
    dst = args.outdir+'/'

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

    config = '/app/densecrf/config_densecrf.txt'
    # t2w_fname = re.sub('T1', 'FLAIR', t1w_fname[0])
    print(t1w_fname[0], t2w_fname[0])
    start_time = time.time()
    denseCRF(case_id[0], t1w_fname[0], t2w_fname, out_shape, config, dst, dst, os.path.join(dst, case_id[0]+"_vnet_maskpred.nii.gz"))
    elapsed_time = time.time() - start_time
    print("=*80")
    print("=> dense 3D-CRF inference time: {} seconds".format(round(elapsed_time,2)))
    print("=*80")


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def find_replace_re(config_tmp, find_str, replace_str):
    with fileinput.FileInput(config_tmp, inplace=True, backup='.bak') as file:
        for line in file:
            print(re.sub(find_str, str(replace_str), line.rstrip(), flags=re.MULTILINE), end='\n')


def denseCRF(id, t1, t2, input_shape, config, in_dir, out_dir, pred_labels):
    X, Y, Z = input_shape
    config_tmp = "/tmp/" + id + "_config_densecrf.txt"
    print(config_tmp)
    subprocess.call(["cp", "-f", config, config_tmp])
    # find and replace placeholder with actual filenames
    find_str = [
                "<ID_PLACEHOLDER>", "<T1_FILE_PLACEHOLDER>", "<FLAIR_FILE_PLACEHOLDER>",
                "<OUTDIR_PLACEHOLDER>", "<PRED_LABELS_PLACEHOLDER>",
                "<X_PLACEHOLDER>", "<Y_PLACEHOLDER>", "<Z_PLACEHOLDER>"
                ]
    replace_str = [
                    str(id), str(t1), str(t2),
                    str(out_dir), str(pred_labels),
                    str(X), str(Y), str(Z)
                    ]
    # config_tmp_replicate = [x for x in [config_tmp] for _ in range(len(find_str))]
    # [print(a,b,c) for a,b,c in zip(config_tmp_replicate, find_str, replace_str)]
    # starmap(find_replace_re, zip(config_tmp_replicate, find_str, replace_str))
    for fs, rs in zip(find_str, replace_str):
        find_replace_re(config_tmp, fs, rs)
    subprocess.call(["/app/dense3dCrf/dense3DCrfInferenceOnNiis", "-c", config_tmp])


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
