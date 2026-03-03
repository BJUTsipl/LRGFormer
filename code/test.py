# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data import ValData
from CS_unet1027 import UNet_emb #UNet
from utilss import validation_PSNR, generate_filelist
import os
# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of dehamer from Guo et al. (2022)')
parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['NH', 'dense', 'indoor','outdoor','our_test','Raindrop'], default='Raindrop')
parser.add_argument('-t', '--test-image-dir', help='test images path', default='./data/classic_test_image/')
parser.add_argument('-c', '--ckpts-dir', help='ckpts path', default='./raindrop-epoch15-0.04094.pt')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
args = parser.parse_args()


val_batch_size = args.val_batch_size
dataset_name = args.dataset_name
# import pdb;pdb.set_trace()
# --- Set dataset-specific hyper-parameters  --- #
if dataset_name == 'NH':
    val_data_dir = './data/valid_NH/'
    ckpts_dir = './ckpts/NH/PSNR2066_SSIM06844.pt'
elif dataset_name == 'dense': 
    val_data_dir = './data/valid_dense/'
    ckpts_dir = './ckpts/dense/PSNR1662_SSIM05602.pt'
elif dataset_name == 'indoor': 
    val_data_dir = './data/valid_indoor/'
    ckpts_dir = './ckpts/indoor/PSNR3663_ssim09881.pt'
elif dataset_name == 'outdoor': 
    val_data_dir = '/media/sipl23/6CEA2B31EA2AF74C/haze_new'
    ckpts_dir = './ckpts/outdoor/dehamer-epoch1-0.02797.pt'
elif dataset_name == 'Raindrop':
    val_data_dir = '/media/HDD-4T/dataset/Weather/Rain/Raindrop/test_b/test_b'
    ckpts_dir = '/media/HDD-4T/mutitask_SF/g4_kxnet1_fft3_Bi4_SGFN_l1/g4_kxnet1_fft3_Bi4_SGFN_l1_derain/raindrop-epoch15-0.04094.pt'
else:
    val_data_dir = args.test_image_dir
    ckpts_dir =  args.ckpts_dir

# prepare .txt file
if not os.path.exists(os.path.join(val_data_dir, 'val_list.txt')):
    generate_filelist(val_data_dir, valid=True)

# --- Gpu device --- #
device_ids =  [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from datasetsio  import TrainDatasetFromFolder4,TrainDatasetFromFolder3,TrainDatasetFromFolder2,TestDatasetFromFolder2
# --- Validation data loader --- #
# val_data_loader = DataLoader(ValData(dataset_name,val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)
val_data_loader = DataLoader(TestDatasetFromFolder2('/media/HDD-4T/dataset/Weather/Rain/Raindrop/test_b/test_b'))
# --- Define the network --- #

# --- Define the network --- #
net = UNet_emb()

  
# --- Multi-GPU --- # 
net = net.to(device)
# net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(ckpts_dir), strict=False)


# --- Use the evaluation model in testing --- #
net.eval() 
print('--- Testing starts! ---') 
start_time = time.time()
validation_PSNR(net , val_data_loader, device, dataset_name, save_tag=True)
end_time = time.time() - start_time
# print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))

print('validation time is {0:.4f}'.format(end_time))
