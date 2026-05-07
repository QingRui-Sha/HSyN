"""
Helper functions from https://github.com/zhangjun001/ICNet.

Some functions has been modified.
"""

import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import itertools
from itertools import product
from scipy.ndimage import zoom

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0]-1)/2)) / (imgshape[0]-1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1]-1)/2)) / (imgshape[1]-1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2]-1)/2)) / (imgshape[2]-1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z-1)
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x-1)

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z-1)
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y-1)
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x-1)

    return flow


def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def crop_center(img, cropx, cropy, cropz):
    x, y, z = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    startz = z//2 - cropz//2
    return img[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]


def load_4D_with_crop(name, cropx, cropy, cropz):
    X = nib.load(name)
    X = X.get_fdata()

    x, y, z = X.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    startz = z//2 - cropz//2

    X = X[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]

    X = np.reshape(X, (1,) + X.shape)
    return X


def load_4D_with_header(name):
    X = nib.load(name)
    X_npy = X.get_fdata()
    X_npy = np.reshape(X_npy, (1,) + X_npy.shape)
    return X_npy, X.header, X.affine


def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,)+(1,)+ X.shape)
    return X


def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min)/(i_max - i_min)
    return norm


def save_img(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_flow(I_img,savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


class Dataset_epoch(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=False):
        'Initialization'
        super(Dataset_epoch, self).__init__()
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])#[...,:64]
        img_B = load_4D(self.index_pair[step][1])#...,:64]

        # img_A = zoom(img_A, (1, 0.5, 0.5, 0.5), order=0)
        # img_B = zoom(img_B, (1, 0.5, 0.5, 0.5), order=0)


        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ndgrid(*args, **kwargs):
    """
    broadcast Tensors on an N-D grid with ij indexing
    uses meshgrid with ij indexing

    Parameters:
        *args: Tensors with rank 1
        **args: "name" (optional)

    Returns:
        A list of Tensors

    """
    return torch.meshgrid(*args, indexing='ij', **kwargs)
def prod_n(lst):
    """
    Alternative to torch.stacking and prod
    """
    prod = lst[0].clone()
    for p in lst[1:]:
        prod *= p
    return prod
def sub2ind2d(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), \
        'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])
    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx


def interpn(vol, loc, interp_method='linear', fill_value=None, device=device):
    vol = vol.to(device)
    if isinstance(loc, (list, tuple)):
        loc = torch.stack(loc, dim=-1).to(device)
    nb_dims = loc.shape[-1]
    input_vol_shape = vol.shape

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = torch.unsqueeze(vol, -1)
    
    # Flatten and float location tensors
    if not loc.dtype.is_floating_point:
        target_loc_dtype = vol.dtype if vol.dtype.is_floating_point else torch.float32
        loc = loc.to(target_loc_dtype)
    elif vol.dtype.is_floating_point and vol.dtype != loc.dtype:
        loc = loc.to(vol.dtype)

    if isinstance(vol.shape, torch.Size):
        vol_shape = list(vol.shape)
    else:
        vol_shape = vol.shape

    max_loc = [d - 1 for d in list(vol.shape)]
    
    if interp_method == 'linear':
        loc0 = loc.floor()

        # Clip values
        clipped_loc = [loc[..., d].clamp(0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [loc0[..., d].clamp(0, max_loc[d]) for d in range(nb_dims)]

        # Get other end of point cube
        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.to(torch.int32) for f in loc0lst], [f.to(torch.int32) for f in loc1]]

        # Compute the difference between the upper value and the original value
        # Differences are basically 1 - (pt - floor(pt))
        #  because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        # Note reverse ordering since weights are inverse of diff.
        weights_loc = [diff_loc1, diff_loc0]
       
        # Go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
    
        for c in cube_pts:
            subs = [locs[c[d]][d] for d in range(nb_dims)]
           
            idx = sub2ind2d(vol_shape[:-1], subs)
            vol_reshape = torch.reshape(vol, [-1, vol_shape[-1]])
            vol_val = vol_reshape[idx.to(torch.int64)]  # torch version of tf.gather()

            # Get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            wt = prod_n(wts_lst)
            wt = wt.unsqueeze(-1)
            # Compute final weighted value for each cube corner
            interp_vol += wt * vol_val

    else:
        assert interp_method == 'nearest', \
            'method should be linear or nearest, got: %s' % interp_method
        roundloc = loc.round().to(torch.int32)
        roundloc = [roundloc[..., d].clamp(0, max_loc[d]) for d in range(nb_dims)]

        idx = sub2ind2d(vol_shape[:-1], roundloc)
        interp_vol = vol.reshape(-1, vol_shape[-1])[idx.long()]

    if fill_value is not None:
        out_type = interp_vol.dtype
        fill_value = torch.tensor(fill_value, dtype=out_type)
        below = [loc[..., d] < 0 for d in range(nb_dims)]
        above = [loc[..., d] > max_loc[d] for d in range(nb_dims)]
        out_of_bounds = torch.any(torch.stack(below + above, dim=-1), dim=-1, keepdim=True)
        interp_vol *= torch.logical_not(out_of_bounds).to(out_type)
        interp_vol += out_of_bounds.to(out_type) * fill_value

    if len(input_vol_shape) == nb_dims:
        assert interp_vol.shape[-1] == 1, 'Something went wrong with interpn channels'
        interp_vol = interp_vol[..., 0]

    return interp_vol

def resize_(vol, zoom_factor, interp_method='linear'):
    """
    if zoom_factor is a list, it will determine the ndims, in which case vol has to be of 
        length ndims of ndims + 1

    if zoom_factor is an integer, then vol must be of length ndims + 1

    If you find this function useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148

    """
    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]

        assert len(vol_shape) in (ndims, ndims + 1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)
    
    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims
    
    # Skip resize for zoom_factor of 1
    if all(z == 1 for z in zoom_factor):
        return vol
    
    if not isinstance(vol_shape[0], int):
        vol_shape = list(vol_shape)
    
    new_shape = [vol_shape[f] * zoom_factor[f] for f in range(ndims)]
    new_shape = [int(f) for f in new_shape]

    lin = [torch.linspace(0., vol_shape[d] - 1., new_shape[d]) for d in range (ndims)]
    grid = ndgrid(*lin)
    grid = [g.to('cuda') for g in grid]

    return interpn(vol, grid, interp_method=interp_method)


def draw_perlin(out_shape,
                scales,
                min_std=0,
                max_std=1,
                modulate=None,
                dtype=torch.float32,
                seed=None,
                device=device):
    '''
    Generate Perlin noise by drawing from Gaussian distributions at different
    resolutions, upsampling and summing. 

    Parameters:
        out_shape: List defining the output shape. In N-dimensional space, it
            should have N+1 elements, the last one being the feature dimension.
        scales: List of relative resolutions at which noise is sampled normally.
            A scale of 2 means half resolution relative to the output shape.
        min_std: Minimum standard deviation (SD) for drawing noise volumes.
        max_std: Maximum SD for drawing noise volumes.
        modulate: Boolean. Whether the SD for each scale is drawn from [0, max_std].
            The argument is deprecated: use min_std instead.
        dtype: Output data type.
        seed: Integer for reproducible randomization. This may only have an
            effect if the function is wrapped in a Lambda layer.
    '''
    out_shape_np = np.asarray(out_shape, dtype=np.int32)
    if isinstance(scales, (int)):
        scales = [scales]

    if not modulate:
        min_std = max_std
    if modulate is not None:
        warnings.warn('Argument modulate to ne.utils.augment.draw_perlin is deprecated '
                      'and will be removed in the future. Use min_std instead.')
        
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(int).max).item()
    rng = torch.Generator(device=device).manual_seed(seed())
    
    out = torch.zeros(out_shape, dtype=dtype, device=device)
    for scale in scales:
        sample_shape = np.ceil(out_shape_np[:-1] / scale)
        sample_shape = np.int32((*sample_shape, out_shape_np[-1]))

        std = torch.empty(size=(), dtype=dtype, device=device).uniform_(min_std, max_std, generator=rng)
        gauss = torch.empty(size=tuple(sample_shape), dtype=dtype, device=device).normal_(std=std, generator=rng)

        zoom = [o / s for o, s in zip(out_shape, sample_shape)]
        out += gauss if scale == 1 else resize_(gauss, zoom[:-1])

    # Transform to Torch format
    indices = list(range(len(out.shape)))
    out = out.permute(-1, *indices[:-1])

    return out

def minmax_norm(x, axis=None):
    """
    Min-max normalize tensor using a safe division.
    Arguments:
        x: Tensor to be normalized.
        axis: Dimensions to reduce during normalization. If None, all axes will be considered,
            treating the input as a single image. To normalize batches or features independently,
            exclude the respective dimensions.
    Returns:
        Normalized tensor.
    """ 
    
    if axis == None:
        # Treated as fattened, 1D tensor
        torchmin = lambda x: torch.min(x)
        torchmax = lambda x: torch.max(x)
    else:
        # Operates on specified axis, and maintain shape
        torchmin = lambda x: torch.min(x, dim=axis, keepdim=True).values
        torchmax = lambda x: torch.max(x, dim=axis, keepdim=True).values

    x_min = torchmin(x)
    x_max = torchmax(x)
    result = torch.where((x_max - x_min) != 0, (x - x_min) / (x_max - x_min), torch.zeros_like(x))
    return result


@torch.no_grad()
def labels_to_image(
    labels,
    out_label_list=None,
    out_shape=None,
    num_chan=1,
    mean_min=None,
    mean_max=None,
    std_min=None,
    std_max=None,
    zero_background=0.2,
    affine_args=None,
    warp_res=[16],
    warp_std=0.5,
    warp_modulate=True,
    bias_res=40,
    bias_std=0.3,
    bias_modulate=True,
    blur_std=1,
    blur_modulate=True,
    normalize=True,
    gamma_std=0.25,
    dc_offset=0,
    one_hot=True,
    seeds={},
    return_vel=False,
    return_def=False,
    device=device
):
    """
    Augment label maps and synthesize images from them.

    Parameters:
        out_label_list (optional): List of labels in the output label maps. If
            a dictionary is passed, it will be used to convert labels, e.g. to
            GM, WM and CSF. All labels not included will be converted to
            background with value 0. If 0 is among the output labels, it will be
            one-hot encoded. Defaults to the input labels.
        out_shape (optional): List of the spatial dimensions of the outputs.
            Inputs will be symmetrically cropped or zero-padded to fit.
            Defaults to the input shape.
        num_chan (optional): Number of image channels to be synthesized.
            Defaults to 1.
        mean_min (optional): List of lower bounds on the means drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            25 for all other labels.
        mean_max (optional): List of upper bounds on the means drawn to generate
            the intensities for each label. Defaults to 225 for each label.
        std_min (optional): List of lower bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            5 for all other labels.
        std_max (optional): List of upper bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 25 for each label.
            25 for all other labels.
        zero_background (float, optional): Probability that the background is set
            to zero. Defaults to 0.2.
        warp_res (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the SVF is drawn.
            Defaults to 16.
        warp_std (float, optional): Upper bound on the SDs used when drawing
            the SVF. Defaults to 0.5.
        warp_modulate (bool, optional): Whether to draw the SVF with random SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        bias_res (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the bias field is
            drawn. Defaults to 40.
        bias_std (float, optional): Upper bound on the SDs used when drawing
            the bias field. Defaults to 0.3.
        bias_modulate (bool, optional): Whether to draw the bias field with
            random SDs. If disabled, each batch will use the maximum SD.
            Defaults to True.
        blur_std (float, optional): Upper bound on the SD of the kernel used
            for Gaussian image blurring. Defaults to 1.
        blur_modulate (bool, optional): Whether to draw random blurring SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        normalize (bool, optional): Whether the image is min-max normalized.
            Defaults to True.
        gamma_std (float, optional): SD of random global intensity
            exponentiation, i.e. gamma augmentation. Defaults to 0.25.
        dc_offset (float, optional): Upper bound on global DC offset drawn and
            added to the image after normalization. Defaults to 0.
        one_hot (bool, optional): Whether output label maps are one-hot encoded.
            Only the specified output labels will be included. Defaults to True.
        seeds (dictionary, optional): Integers for reproducible randomization.
        return_vel (bool, optional): Whether to append the half-resolution SVF
            to the model outputs. Defaults to False.
        return_def (bool, optional): Whether to append the combined displacement
            field to the model outputs. Defaults to False.
    """
    np_rng = np.random.default_rng(None)
    default_seed = lambda: np_rng.integers(np.iinfo(int).max).item()
    rng = lambda x: torch.Generator(device=device).manual_seed(x)

    batch_size = 1
    num_dim = len(labels.shape)
    in_shape = labels.shape
    if out_shape is None:
        out_shape = in_shape
    in_shape, out_shape = map(np.asarray, (in_shape, out_shape))

    # Add new axes, Torch format
    labels = labels.unsqueeze(0).unsqueeze(1)
    labels = labels.expand(batch_size, -1, *[-1] * num_dim)

    # Transform labels into [0, 1, ..., N-1].
    labels = labels.to(dtype=torch.int32, device=device)
    in_label_list = labels.unique()
    num_in_labels = len(in_label_list)

    in_lut = torch.zeros(size=(torch.max(in_label_list) + 1,), dtype=torch.int32, device=device)
    for i, lab in enumerate(in_label_list):
        in_lut[lab] = i
    # print(labels,'labels')
    labels = in_lut[labels.long()] # tf.gather(in_lut, indices=labels)
    labels = torch_to_tf(labels)  # TF format

    # labels = torch.ones(labels.shape, device=device)        #remove later

    def_field = None
    vel_field = None
    if warp_std > 0:
        # Velocity field.
        vel_shape = (*out_shape // 2, num_dim)
        vel_scale = np.asarray(warp_res) / 2
        vel_draw = lambda: draw_perlin(
            vel_shape, scales=vel_scale,
            min_std=0 if warp_modulate else warp_std, max_std=warp_std,
            seed=seeds.get('warp')
        )
        # One per batch.
        vel_field = torch.stack([vel_draw() for _ in labels])
        vel_field = torch_to_tf(vel_field)
        # Deformation field.
        def_field = layers.VecInt(int_steps=5)(vel_field)
        def_field = layers.RescaleValues(2)(def_field)
        def_field = layers.Resize(2, interp_method='linear')(def_field)
        # Resampling.
        labels = layers.SpatialTransformer(interp_method='nearest', fill_value=0)([labels, def_field])

    # Affine transformations
    if affine_args is not None:
        labels = tf_to_torch(labels)    # torchvision operates on torch format
        if not isinstance(affine_args, dict):
            warnings.warn("Argument affine_args must be a dictionary to apply affine transformations")
        rotate = affine_args.get("rotate", 0)
        translate = affine_args.get("translate", None)
        scale = affine_args.get("scale", None)
        shear = affine_args.get("shear", None)
        aff_transformer = RandomAffine(degrees=rotate, translate=translate, scale=scale, shear=shear)
        labels = aff_transformer(labels)
        labels = torch_to_tf(labels)
    
    labels = labels.to(torch.int32)
    # Intensity means and standard deviations for synthetic image
    if mean_min is None:
        mean_min = [0] + [25] * (num_in_labels - 1)
    if mean_max is None:
        mean_max = [225] * num_in_labels
    if std_min is None:
        std_min = [0] + [5] * (num_in_labels - 1)
    if std_max is None:
        std_max = [25] * num_in_labels
    as_torch_tensor = lambda x: torch.as_tensor(x, device=device)
    m0, m1, s0, s1 = map(as_torch_tensor, (mean_min, mean_max, std_min, std_max))

    mean = torch.rand(
        size=(batch_size, num_chan, num_in_labels),
        generator=rng(seeds.get('mean', default_seed())),
        device=device,
    )
    mean = m0 + (m1 - m0) * mean

    std = torch.rand(
        size=(batch_size, num_chan, num_in_labels),
        generator=rng(seeds.get('std', default_seed())),
        device=device,
    )
    std = s0 + (s1 - s0) * std

    # Synthetic image.
    image = torch.empty(size=labels.shape, device=device).normal_(generator=rng(seeds.get('noise', default_seed())))
    indices = torch.concat([labels + i * num_in_labels for i in range(num_chan)], dim=-1)
    gather = lambda x: torch.reshape(x[0], (-1,))[x[1].long()]
    mean = gather([mean, indices])
    std = gather([std, indices])
    image = image * std + mean

    # Zero background.
    if zero_background > 0:
        rand_flip = torch.rand(
            size=(batch_size, *[1] * num_dim, num_chan),
            generator=rng(seeds.get('background', default_seed())),
            device=device,
        )
        rand_flip = torch.lt(rand_flip, zero_background)    # tf.less
        image *= 1. - torch.logical_and(labels == 0, rand_flip).to(image.dtype)

    # Blur.
    if blur_std > 0:
        if num_dim==2:
            kernels = utils.gaussian_kernel(
                [blur_std] * num_dim, separate=True, random=blur_modulate,
                dtype=image.dtype, seed=seeds.get('blur'),
            )
            print(image.shape,len(kernels),kernels[0].shape)
            image = utils.separable_conv(image, kernels, batched=True)#!!!!!!!!!!!!
        else:
            gaussian_kernel = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
            gaussian_kernel.weight.data = torch.ones(1, 1, 3, 3, 3,dtype=torch.float).cuda()  # 使用3x3x3的卷积核
            gaussian_kernel.weight.data *= blur_std  # 根据标准差调整卷积核的权重
            # print(kernels)
            # print(image.shape)
            image=torch.squeeze(image,dim=4)
            image=torch.unsqueeze(image,dim=0)
            # print(image.dtype,gaussian_kernel.weight.data.dtype)
            
            # image=gaussian_kernel(image)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            image=torch.squeeze(image,dim=1)
            image=torch.unsqueeze(image,dim=4)
            # print(image.shape)
            # image = nnf.conv3d(image, torch.tensor(kernels))#, batched=True)

    # Bias field.
    if bias_std > 0:
        bias_shape = (*out_shape, 1)
        bias_draw = lambda: draw_perlin(
            bias_shape, scales=bias_res, seed=seeds.get('bias'),
            min_std=0 if bias_modulate else bias_std, 
            max_std=bias_std, device=device
        )
        bias_field = torch.stack([bias_draw() for _ in labels])
        bias_field = torch_to_tf(bias_field)  # TF format
        # print(bias_field.shape,image.shape)
        image *= torch.exp(bias_field)

    # Intensity manipulations.
    image = torch.clip(image, min=0, max=255)
    if normalize:
        image = torch.stack([minmax_norm(batch) for batch in image])  
    if gamma_std > 0:
        gamma = torch.empty(size=(batch_size, *[1] * num_dim, num_chan), device=device)
        gamma = gamma.normal_(std=gamma_std, generator=rng(seeds.get('gamma', default_seed())))
        image = torch.pow(image, torch.exp(gamma))
    if dc_offset > 0:
        offset = torch.empty(size=(batch_size, *[1] * num_dim, num_chan), device=device)
        offset = offset.uniform_(0, dc_offset, generator=rng(seeds.get('dc_offset', default_seed())))
        image += offset

    image = tf_to_torch(image)
    
    # Lookup table for converting the index labels back to the original values,
    # setting unwanted labels to background. If the output labels are provided
    # as a dictionary, it can be used e.g. to convert labels to GM, WM, CSF.
    if out_label_list is None:
        out_label_list = in_label_list
    if isinstance(out_label_list, (tuple, list, torch.Tensor)):
        out_label_list = {lab.item(): lab for lab in out_label_list}

    out_lut = torch.zeros((num_in_labels,), dtype=torch.int32)
    for i, lab in enumerate(in_label_list):
        if lab.item() in out_label_list:
            out_lut[i] = out_label_list[lab.item()]

    # For one-hot encoding, update the lookup table such that the M desired
    # output labels are rebased into the interval [0, M-1[. If the background
    # with value 0 is not part of the output labels, set it to -1 to remove it
    # from the one-hot maps.
    if one_hot:
        hot_label_list = torch.tensor(list(out_label_list.values())).unique() # Sorted.
        hot_lut = torch.full((hot_label_list[-1] + 1,), fill_value=-1, dtype=torch.int32, device=device)
        for i, lab in enumerate(hot_label_list):
            hot_lut[lab] = i
        out_lut = hot_lut[out_lut.long()]

    # Convert indices to output labels only once.
    labels = out_lut[labels.long()]
    if one_hot:
        labels = nnf.one_hot(labels.to(torch.int64), num_classes=len(hot_label_list))
        labels = tf_to_torch(labels.squeeze(-2))

    # Remove batch_size
    if vel_field is not None:
        all_outputs = [image, labels, vel_field, def_field]
        image, labels, vel_field, def_field = [i.squeeze(0) for i in all_outputs]
    else:
        all_outputs = [image, labels]
        image, labels = [i.squeeze(0) for i in all_outputs]
    
    
    outputs = {'image': image, 'label': labels}
    if return_vel:
        outputs['vel'] = vel_field
    if return_def:
        outputs['def'] =  def_field

    return outputs
def torch_to_tf(inp: torch.Tensor):
    indices = np.arange(inp.ndim)
    return inp.permute(0, *indices[2:], 1)


def tf_to_torch(inp: torch.Tensor): 
    indices = np.arange(inp.ndim)
    return inp.permute(0, -1, *indices[1:-1])
def merge_label(im1):
    label=[13,    26,     25,       20,   24,      28,  33,       29,      27,   22,       30,   11,    12,      31,    21,       35]
    
    im1[im1==7]=26
    
    im1[im1==6]=25
    
    im1[im1==1]=20
    
    im1[im1==3]=22
    
    im1[im1==5]=24
    
    im1[im1==9]=28
    
    im1[im1==8]=27
    
    im1[im1==10]=29
    
    im1[im1==14]=30
    
    im1[im1==15]=31
    
    im1[im1==2]=21
    
    im1[im1==17]=33
    
    im1[im1==19]=35
    
    return im1





class Predict_dataset_mutil_fixed(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, fixed_label_list_4, move_label_list_4, norm=False):
        super(Predict_dataset_mutil_fixed, self).__init__()
        self.image_list=list(product(fixed_list,move_list))
        self.seg_list=list(product(fixed_label_list,move_label_list))
        self.seg_list_4=list(product(fixed_label_list_4,move_label_list_4))
        
        # self.fixed_list = fixed_list
        # self.move_list = move_list
        # self.fixed_label_list = fixed_label_list
        # self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.image_list[index][0])
        moved_img = load_4D(self.image_list[index][1])
        fixed_label = load_4D(self.seg_list[index][0])
        moved_label = load_4D(self.seg_list[index][1])
        fixed_label_4 = load_4D(self.seg_list_4[index][0])
        moved_label_4 = load_4D(self.seg_list_4[index][1])

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_img = torch.from_numpy(fixed_img)#[...,:64]
        moved_img = torch.from_numpy(moved_img)#[...,:64]
        fixed_label = torch.from_numpy(fixed_label)#[...,:64]
        moved_label = torch.from_numpy(moved_label)#[...,:64]

        fixed_label_4 = torch.from_numpy(fixed_label_4)#[...,:64]
        moved_label_4 = torch.from_numpy(moved_label_4)#[...,:64]

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': fixed_label.float(), 'move_label': moved_label.float(),
                   'fixed_label_4': fixed_label_4.float(), 'move_label_4': moved_label_4.float(), 'index': index}
        return output

