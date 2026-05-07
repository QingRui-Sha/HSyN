import os,sys
import random
import argparse
import time
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

import glob
import voxelmorph as vxm  



parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--datapath', default='')

parser.add_argument('--model-dir', default='./models',
                    help='model and results output directory (default: models)')
parser.add_argument('--losses-names', default=['sim','smooth','mid_dis'])
parser.add_argument('--inverse_losses-names', default=['sim','smooth'])                   

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load_model', help='optional model file to initialize with')
parser.add_argument('--load_model_InverseNet', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--weight-smooth', type=float,  default=1)
parser.add_argument('--weight-image-sim', type=float, default=1)
parser.add_argument('--weight-tre', type=float,  default=0)
parser.add_argument('--weight-inverse-smooth', type=float,  default=0.02)
parser.add_argument('--weight-inverse-mse', type=float, default=1)
parser.add_argument('--seg-number', type=float, default=30)
parser.add_argument('--threshold', type=float, default=0.01)
parser.add_argument("--weight-mid-dis", type=float, default=0.1,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--log-name", type=str,
                    dest="log_name", default='val.txt')
parser.add_argument("--lv1", type=int,default=300,
                    help="number of lvl1 iterations")
parser.add_argument("--lv2", type=int,default=600,
                    help="number of lvl2 iterations")
parser.add_argument("--lv3", type=int,default=900,
                    help="number of lvl3 iterations")
parser.add_argument("--unfreeze-step", type=int,
                    dest="unfreeze_step", default=40)
parser.add_argument("--threshold-mse", type=bool, default=False)

args = parser.parse_args()
writer = SummaryWriter(log_dir=os.path.join(
        args.model_dir, 'tensorboard'))
bidir = args.bidir

log = open(os.path.join(
        args.model_dir, 'log.txt'), "w", buffering=1)  
sys.stdout = log
sys.stderr = log
# load and prepare training data


all_norm = sorted(glob.glob(args.datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
all_seg35 = sorted(glob.glob(args.datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))
all_seg4  = sorted(glob.glob(args.datapath + '/OASIS_OAS1_*_MR1/aligned_seg4.nii.gz'))


assert len(all_norm) == len(all_seg35) == len(all_seg4)

N = len(all_norm)

random.seed(42)
idx = list(range(N))
random.shuffle(idx)


train_idx = idx[:255]
fixed_idx = idx[275:280]
eval_idx  = idx[280:]


names = [all_norm[i] for i in train_idx]

fixed_img   = [all_norm[i]  for i in fixed_idx]
fixed_label = [all_seg35[i] for i in fixed_idx]
fixed_label_4 = [all_seg4[i] for i in fixed_idx]

imgs      = [all_norm[i]  for i in eval_idx]
labels    = [all_seg35[i] for i in eval_idx]
labels_4  = [all_seg4[i]  for i in eval_idx]


training_generator = Data.DataLoader(vxm.functions.Dataset_epoch(names, norm=True), batch_size=args.batch_size,
                                         shuffle=True, num_workers=4)


prefetcher = vxm.datasets.data_prefetcher(training_generator)
inputs = prefetcher.next()


count_parameters=vxm.py.utils.count_parameters

# extract shape from sampled input
inshape = (160, 192,224)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(args.model_dir,args.log_name), "a") as log:
    log.write("Training log for oasis:\n")
# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda:0'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense_dis_pad_double_head.load(args.load_model, device)
    model_InverseNet = vxm.networks.VxmDense_InverseNet_displacement_resize_pad_only_flow.load(args.load_model_InverseNet, device)
else: 
    # otherwise configure new model
    model = vxm.networks.VxmDense_dis_pad_double_head(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
    )
    model_InverseNet = vxm.networks.VxmDense_InverseNet_displacement_resize_pad_only_flow(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model_InverseNet.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
num_para=count_parameters(model)
num_para2=count_parameters(model_InverseNet)
# print("Total number of parameters: ",num_para,num_para2)


# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer_InverseNet = torch.optim.Adam(model_InverseNet.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC_()
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# prepare deformation loss
smooth_loss_func=vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss
tre_loss_func=vxm.losses.Tre().loss
bidir_tre_loss_func=vxm.losses.Tre_Bidir().loss
dice_loss_func=vxm.losses.Dice().loss
mse_loss_func = vxm.losses.MSE().loss
mid_dis_loss_func = vxm.losses.Mid_dis().loss

if args.threshold_mse:
    mse_threshold_loss_func = vxm.losses.MSE_threshold().loss
else:
    print('MSE')
    mse_threshold_loss_func = vxm.losses.MSE().loss
if args.seg_number==24:
    mutil_dice=vxm.losses.mutil_dice24
elif args.seg_number==35:
    mutil_dice=vxm.losses.mutil_dice35
elif args.seg_number==30:
    mutil_dice=vxm.losses.mutil_dice30
best_dsc=0




        

    
    

model.eval()
model_InverseNet.eval()

jac_list=[]
tre_list=[]
dice_list=[]
total_loss_list=[]
inverse_mse_list=[]
inverse_jac_list=[]
inverse_total_loss_list=[]
final_jac_list=[]
final_tre_list=[]
final_dice_list=[]
train_level=3
with torch.no_grad():
    # Validation
    

    valid_generator = Data.DataLoader(vxm.functions.Predict_dataset_mutil_fixed(fixed_img, imgs, fixed_label, labels, fixed_label_4, labels_4, norm=True),
                                        batch_size=1,
                                        shuffle=False, num_workers=4)

    
    for batch_idx, data in enumerate(valid_generator):
        moving_image, fixed_image, moving_seg, fixed_seg,moving_seg_4, fixed_seg_4 = data['move'].to(device), data['fixed'].to(device), data['move_label'].to(
            device), data['fixed_label'].to(device), data['move_label_4'].to(
            device), data['fixed_label_4'].to(device)
        
        flow1,flow2,source, target,warped_source, warped_target=model(moving_image,fixed_image,level=train_level)
        if train_level==1:
            flow=model.fullsize_4(flow1)
            f2m_flow=model.fullsize_4(flow2)
        elif train_level==2:
            flow=model.fullsize(flow1)
            f2m_flow=model.fullsize(flow2)
        else:
            flow=flow1
            f2m_flow=flow2
        small_v=flow
        f2m_small_v=f2m_flow
        warped_image=model.transformer(moving_image,flow)
        f2m_warped_image=model.transformer(fixed_image,f2m_flow)

        warped_seg=model.transformer(moving_seg,flow,mode='nearest')
        warped_seg_4=model.transformer(moving_seg_4,flow,mode='nearest')
        f2m_warped_seg=model.transformer(fixed_seg,f2m_flow,mode='nearest')
        f2m_warped_seg_4=model.transformer(fixed_seg_4,f2m_flow,mode='nearest')

        
        # neg_f2m_flow=model.neg_field(f2m_small_v)
        # neg_flow=model.neg_field(small_v)
        flow_identify,small_inverse_flow,neg_flow=model_InverseNet(flow)
        f2m_flow_identify,small_inverse_f2m_flow,neg_f2m_flow=model_InverseNet(f2m_flow)
        
        final_flow=model.composition_transformer(flow,neg_f2m_flow)
        
        jac=vxm.py.utils.jacobian_determinant(final_flow.permute(0,2,3,4,1).detach().cpu().numpy().squeeze())[...,3:-3,3:-3,3:-3]
        # f2m_jac=vxm.py.utils.jacobian_determinant(f2m_flow.permute(0,2,3,4,1).detach().cpu().numpy().squeeze())
        
        #val loss
        image_sim_loss = image_loss_func(f2m_warped_image, warped_image) * args.weight_image_sim
        
        smooth_loss = (smooth_loss_func(small_v, small_v)+smooth_loss_func(f2m_small_v, f2m_small_v)) * args.weight_smooth
        mid_dis_loss=mid_dis_loss_func(flow,f2m_flow)*args.weight_mid_dis

        loss=image_sim_loss+smooth_loss+mid_dis_loss
        
        
        dice_=mutil_dice(f2m_warped_seg,warped_seg,f2m_warped_seg_4,warped_seg_4)
        
        
        #record loss
        
        jac_list.append((np.sum(jac<0))/np.prod(jac.shape))
        dice_list.append(dice_.item())
        total_loss_list.append(loss.item())
        

        
        

        #---------------------------
        #model InverseNet
        #---------------------------
        y_identify,small_inverse_v,inverse_flow=model_InverseNet(f2m_flow)
        #extra val loss
        inverse_mse_=mse_loss_func(y_identify,torch.zeros_like(y_identify))
        inverse_jac=vxm.py.utils.jacobian_determinant(inverse_flow.permute(0,2,3,4,1).detach().cpu().numpy().squeeze())[...,3:-3,3:-3,3:-3]
        #val loss
        inverse_loss = 0
        
        
        # inverse_mse_loss=mse_threshold_loss_func(y_identify,torch.zeros_like(y_identify),args.threshold)*args.weight_inverse_mse
        inverse_mse_loss=mse_threshold_loss_func(y_identify,torch.zeros_like(y_identify))*args.weight_inverse_mse
        inverse_smooth_loss = smooth_loss_func(small_inverse_v, small_inverse_v) * args.weight_inverse_smooth
        
        inverse_loss=inverse_mse_loss+inverse_smooth_loss
        
        #record loss
        inverse_mse_list.append(inverse_mse_.item())
        inverse_jac_list.append(np.sum(inverse_jac<0)/np.prod(inverse_jac.shape))
        inverse_total_loss_list.append(inverse_loss.item())
        
        #final flow
        
        finel2_warped_image=model.transformer(model.transformer(moving_image,flow),neg_f2m_flow)
        image_sim_loss__2 = image_loss_func(fixed_image, finel2_warped_image) 
        finel_warped_image=model.transformer(moving_image,final_flow)
        image_sim_loss__ = image_loss_func(fixed_image, finel_warped_image) 
        final_warped_seg=model.transformer(moving_seg,final_flow,mode='nearest')
        final_warped_seg_4=model.transformer(moving_seg_4,final_flow,mode='nearest')
        dice_final=mutil_dice(final_warped_seg,fixed_seg,final_warped_seg_4,fixed_seg_4)
        final2_warped_seg=model.transformer(model.transformer(moving_seg,flow,mode='nearest'),f2m_flow,mode='nearest')
        final2_warped_seg_4=model.transformer(model.transformer(moving_seg_4,flow,mode='nearest'),f2m_flow,mode='nearest')
        dice_final2=mutil_dice(final2_warped_seg,fixed_seg,final2_warped_seg_4,fixed_seg_4)
        

        acc_neg=mse_loss_func(model.composition_transformer(flow,neg_flow),torch.zeros_like(flow))
        acc_neg_f2m=mse_loss_func(model.composition_transformer(f2m_flow,neg_f2m_flow),torch.zeros_like(flow))
        print(batch_idx,'acc_neg %.6e,acc_f2m_neg %.6e,dice_mid:%.6e,dice_final2:%.6e,dice_final:%.6e,sim_mid %.6e,sim_final2 %.6e,sim_final %.6e'%(
            acc_neg.item(),acc_neg_f2m.item(),dice_.item(),dice_final2.item(),dice_final.item(),
            image_sim_loss.item(),image_sim_loss__2.item(),image_sim_loss__.item()))
        
        
        
        final_dice_list.append(dice_final.item())
        # break
epoch=0
best_dsc = max(np.mean(final_dice_list), best_dsc)
model.save(os.path.join(model_dir, '%04d_dsc%.4f_best_dsc:%.4f.pt' % (epoch,np.mean(final_dice_list),best_dsc)))
model_InverseNet.save(os.path.join(model_dir, '%04d_dsc%.4f_best_dsc:%.4f_InverseNet.pt' % (epoch,np.mean(final_dice_list),best_dsc)))

writer.add_scalar('losses_info_val_final_jac', np.mean(jac_list), epoch)
writer.add_scalar('losses_info_val_mid_dice', np.mean(dice_list), epoch)
writer.add_scalar('total_loss_info_val', np.mean(total_loss_list), epoch)
writer.add_scalar('losses_info_val_inverse_mse', np.mean(inverse_mse_list), epoch)
writer.add_scalar('losses_info_val_inverse_jac', np.mean(inverse_jac_list), epoch)
writer.add_scalar('inverse_total_loss_info_val', np.mean(inverse_total_loss_list), epoch)
writer.add_scalar('losses_info_val_final_dice', np.mean(final_dice_list), epoch)
del warped_image,small_v,flow,f2m_warped_image,f2m_small_v,f2m_flow,y_identify,small_inverse_v,inverse_flow
# print epoch info
epoch_info = 'Test Epoch %d/%d' % (epoch + 1, args.epochs)
loss_info = 'total:%.4e mid_dice:%.4e,final_dice:%.4e,final_jac:%.4e' % (np.mean(total_loss_list),  np.mean(dice_list),np.mean(final_dice_list),np.mean(jac_list))
# print(' - '.join((epoch_info,loss_info)), flush=True)
with open(os.path.join(args.model_dir,args.log_name), "a") as log:
    log.write(str(epoch)+','+str(np.mean(final_dice_list))+','+str(np.mean(jac_list))+'\n')


print(' - '.join((epoch_info,loss_info)), flush=True)








