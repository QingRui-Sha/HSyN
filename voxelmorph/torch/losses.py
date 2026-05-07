import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class NCC_(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC_, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
class MSE_threshold:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred,threshold):
        error=(y_true - y_pred) ** 2
        ret=torch.mean(error[error>=threshold])
        if torch.isnan(ret):
            return torch.mean(error)
        return torch.mean(error[error>=threshold])+torch.mean(error)

class Mid_dis:
    """
    Mean squared error loss.
    """

    def loss(self, flow1, flow2):
        return torch.mean((flow1+flow2) ** 2)
class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice
def mutil_dice35(im1, atlas,im1_4, atlas_4):
    unique_class = torch.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0):
            continue

        sub_dice = torch.sum(atlas[im1 == i] == i) * 2.0 / (torch.sum(im1 == i) + torch.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice/num_count

def mutil_dice24(im1, atlas,im1_4, atlas_4,verbose=False):
    unique_class = torch.unique(atlas)
    dice = 0
    num_count = 0
    # label=[13,   7,26,   6,25,  3,22,   5,24,    9,28,  8,27,   10,29,  14,30,   11,   12,   15,31,    2,21]
    label=[13,   26,   25,  22,   24,    28,  27,   29,  30,   11,   12,   31,    21]
    
    im1[im1==7]=26
    atlas[atlas==7]=26
    im1[im1==6]=25
    atlas[atlas==6]=25
    im1[im1==3]=22
    atlas[atlas==3]=22
    im1[im1==5]=24
    atlas[atlas==5]=24
    im1[im1==9]=28
    atlas[atlas==9]=28
    im1[im1==8]=27
    atlas[atlas==8]=27
    im1[im1==10]=29
    atlas[atlas==10]=29
    im1[im1==14]=30
    atlas[atlas==14]=30
    im1[im1==15]=31
    atlas[atlas==15]=31
    im1[im1==2]=21
    atlas[atlas==2]=21
    dice_label=[]
    dice_list=[]
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0) or (i not in label):
            continue

        sub_dice = torch.sum(atlas[im1 == i] == i) * 2.0 / (torch.sum(im1 == i) + torch.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        dice_label.append(i.item())
        dice_list.append(sub_dice.item())
    if ((im1_4==4).sum()>0) and ((atlas_4==4).sum()>0) :
            
        sub_dice = torch.sum(atlas_4[im1_4 == 4] == 4) * 2.0 / (torch.sum(im1_4 == 4) + torch.sum(atlas_4 == 4))
        dice += sub_dice
        num_count += 1
        dice_label.append(4)
        dice_list.append(sub_dice.item())
    if verbose:
        print(dice_label,dice_list,len(dice_label),np.mean(dice_list))
    return dice/num_count

def mutil_dice32(im1, atlas,im1_4, atlas_4,verbose=False):
    unique_class = torch.unique(atlas)
    dice = 0
    num_count = 0
    # label=[13,   7,26,   6,25,  3,22,   5,24,    9,28,  8,27,   10,29,  14,30,   11,   12,   15,31,    2,21]
    #label=[13,    26,     25,  22,   24,    28,  27,   29,  30,   11,   12,   31,    21]

    label=[13,   7,26,   6,25,   1,20, 5,24,     9,28,  17,33,  10,29,  8,27,  3,22,   14,30,   11,    12,   15,31,   2,21,   19,35]
    # label=[13,    26,     25,       20,   24,      28,  33,       29,      27,   22,       30,   11,    12,      31,    21,       35]
    
    dice_label=[]
    dice_list=[]
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0) or (i not in label):
            continue

        sub_dice = torch.sum(atlas[im1 == i] == i) * 2.0 / (torch.sum(im1 == i) + torch.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        dice_label.append(i.item())
        dice_list.append(sub_dice.item())
    if ((im1_4==4).sum()>0) and ((atlas_4==4).sum()>0) :
            
        sub_dice = torch.sum(atlas_4[im1_4 == 4] == 4) * 2.0 / (torch.sum(im1_4 == 4) + torch.sum(atlas_4 == 4))
        dice += sub_dice
        num_count += 1
        dice_label.append(4)
        dice_list.append(sub_dice.item())
    if verbose:
        print(dice_label,dice_list,len(dice_label),np.mean(dice_list))
    return dice/num_count
def mutil_dice30(im1, atlas,im1_4, atlas_4,verbose=False):
    unique_class = torch.unique(atlas)
    dice = 0
    num_count = 0
    # label=[13,   7,26,   6,25,  3,22,   5,24,    9,28,  8,27,   10,29,  14,30,   11,   12,   15,31,    2,21]
    #label=[13,    26,     25,  22,   24,    28,  27,   29,  30,   11,   12,   31,    21]

    #label=[13,   7,26,   6,25,   1,20, 5,24,     9,28,  17,33,  10,29,  8,27,  3,22,   14,30,   11,    12,   15,31,   2,21,   19,35]
    label=[13,    26,     25,       20,   24,      28,  33,       29,      27,   22,       30,   11,    12,      31,    21,       35]
    
    im1[im1==7]=26
    atlas[atlas==7]=26
    im1[im1==6]=25
    atlas[atlas==6]=25
    im1[im1==1]=20
    atlas[atlas==1]=20
    im1[im1==3]=22
    atlas[atlas==3]=22
    im1[im1==5]=24
    atlas[atlas==5]=24
    im1[im1==9]=28
    atlas[atlas==9]=28
    im1[im1==8]=27
    atlas[atlas==8]=27
    im1[im1==10]=29
    atlas[atlas==10]=29
    im1[im1==14]=30
    atlas[atlas==14]=30
    im1[im1==15]=31
    atlas[atlas==15]=31
    im1[im1==2]=21
    atlas[atlas==2]=21
    im1[im1==17]=33
    atlas[atlas==17]=33
    im1[im1==19]=35
    atlas[atlas==19]=35
    dice_label=[]
    dice_list=[]
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0) or (i not in label):
            continue

        sub_dice = torch.sum(atlas[im1 == i] == i) * 2.0 / (torch.sum(im1 == i) + torch.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        dice_label.append(i.item())
        dice_list.append(sub_dice.item())
    if ((im1_4==4).sum()>0) and ((atlas_4==4).sum()>0) :
            
        sub_dice = torch.sum(atlas_4[im1_4 == 4] == 4) * 2.0 / (torch.sum(im1_4 == 4) + torch.sum(atlas_4 == 4))
        dice += sub_dice
        num_count += 1
        dice_label.append(4)
        dice_list.append(sub_dice.item())
    if verbose:
        print(dice_label,dice_list,len(dice_label),np.mean(dice_list))
    return dice/num_count

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
class Tre:
    """
    tre for keypoints
    """
    def loss(self,flow,fixed_kp,moving_kp,grid_tre,loss=True,threshold=1000):
        assert flow.shape[1]==3
        assert fixed_kp.shape[3]==3
        assert fixed_kp.shape[0]==1
        assert fixed_kp.shape[1]==1

        
        shape = flow.shape[2:]
        fixed_kp_normal=torch.zeros_like(fixed_kp)
        for i in range(len(shape)):
            fixed_kp_normal[...,i] = 2 * (fixed_kp[...,i] / (shape[i] - 1) - 0.5)
        
        
        # grid=self.grid
        
        fixed_kp_normal=fixed_kp_normal[..., [2, 1, 0]]
        
        # print(flow.shape,self.reg_net.transformer.grid.shape,'flow.shape,self.reg_net.transformer.grid.shape')
        warped_fixed_kp= F.grid_sample(flow+grid_tre,fixed_kp_normal.unsqueeze(0),align_corners=True,mode='bilinear')
        
        warped_fixed_kp=warped_fixed_kp.permute(0,2,3,4,1)[0,...]
        if not loss:
            return warped_fixed_kp
            # np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)
        # error=().pow(2).sqrt()
        error=torch.linalg.norm((warped_fixed_kp.squeeze()-moving_kp.squeeze())*torch.tensor([1.5,1.5,1.5]).cuda(),axis=1)
        # print(error.shape,warped_fixed_kp[0,0,:10,:],fixed_kp[0,0,:10,:],moving_kp[0,0,:10,:])
        # print(error[:10])
        # print(torch.mean(error[error<=threshold]).item(),torch.mean(error).item())
        error=torch.mean(error[error<=threshold])
        
        
        return error

class Tre_Bidir:
    """
    Bidirectional tre for keypoints
    """
    def loss(self,neg_flow,neg_f2m_flow,fixed_kp,moving_kp,grid_tre,loss=True,threshold=1000):
        assert neg_flow.shape[1]==3
        assert fixed_kp.shape[3]==3
        assert fixed_kp.shape[0]==1
        assert fixed_kp.shape[1]==1

        
        shape = neg_flow.shape[2:]
        #m2f------------------------
        fixed_kp_normal=torch.zeros_like(fixed_kp)
        for i in range(len(shape)):
            fixed_kp_normal[...,i] = 2 * (fixed_kp[...,i] / (shape[i] - 1) - 0.5)
        
        
        
        fixed_kp_normal=fixed_kp_normal[..., [2, 1, 0]]
        
        # print(flow.shape,self.reg_net.transformer.grid.shape,'flow.shape,self.reg_net.transformer.grid.shape')
        warped_fixed_kp= F.grid_sample(neg_f2m_flow+grid_tre,fixed_kp_normal.unsqueeze(0),align_corners=True,mode='bilinear')
        
        warped_fixed_kp=warped_fixed_kp.permute(0,2,3,4,1)[0,...]

        #f2m----------------------
        moving_kp_normal=torch.zeros_like(moving_kp)
        for i in range(len(shape)):
            moving_kp_normal[...,i] = 2 * (moving_kp[...,i] / (shape[i] - 1) - 0.5)
        
        
        # grid=self.grid
        
        moving_kp_normal=moving_kp_normal[..., [2, 1, 0]]
        
        # print(flow.shape,self.reg_net.transformer.grid.shape,'flow.shape,self.reg_net.transformer.grid.shape')
        warped_moving_kp= F.grid_sample(neg_flow+grid_tre,moving_kp_normal.unsqueeze(0),align_corners=True,mode='bilinear')
        
        warped_moving_kp=warped_moving_kp.permute(0,2,3,4,1)[0,...]
        
        if not loss:
            return warped_fixed_kp,warped_moving_kp
            # np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)
        # error=().pow(2).sqrt()
        error=torch.linalg.norm((warped_fixed_kp.squeeze()-warped_moving_kp.squeeze())*torch.tensor([1.5,1.5,1.5]).cuda(),axis=1)
        # print(error.shape,warped_fixed_kp[0,0,:10,:],fixed_kp[0,0,:10,:],moving_kp[0,0,:10,:])
        # print(error[:10])
        # print(error.shape,'error.shape')
        error=torch.mean(error[error<=threshold])
        
        
        return error
