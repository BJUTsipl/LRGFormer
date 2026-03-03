import glob
import random
import os
from torch import cat
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F  # 放在文件顶端的 imports 区
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose,ToPILImage, RandomCrop, CenterCrop, Resize  ,ToTensor, Normalize
import torchvision.transforms as transforms
import  natsort
#import  h5py
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','.bmp','.BMP'])


def calculate_valid_crop_size(crop_size):
    return crop_size



def train_h_transform(crop_size):
    return Compose([
        # RandomCrop(crop_size),
        # CenterCrop(crop_size),
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

def train_s_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        # RandomCrop(crop_size),
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    ])

def train_trans_transform(crop_size):
    return Compose([
        # CenterCrop(crop_size),
        RandomCrop(crop_size),
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
def test_transform():
    return Compose([
        CenterCrop(256),
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Normalize((0.64, 0.6, 0.58), (0.14, 0.15, 0.152)),

    ])

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='OTS_B'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/hazy' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/clear' % mode) + '/*.*'))


        #self.files_A = sorted(glob.glob(os.path.join(root, '%s/trainA' % mode) + '/*.*'))
        #self.files_B = sorted(glob.glob(os.path.join(root, '%s/trainB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset1(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='real'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/gt' % mode) + '/*.*'))#testA
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/hazy' % mode) + '/*.*'))#testB


        #self.files_A = sorted(glob.glob(os.path.join(root, '%s/trainA' % mode) + '/*.*'))
        #self.files_B = sorted(glob.glob(os.path.join(root, '%s/trainB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset2(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='test-rrrrrr'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/h' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/t' % mode) + '/*.*'))


        #self.files_A = sorted(glob.glob(os.path.join(root, '%s/trainA' % mode) + '/*.*'))
        #self.files_B = sorted(glob.glob(os.path.join(root, '%s/trainB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/testA'#'/hazy'
        self.s_path = dataset_dir + '/testB'#'/clear'
        #self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if is_image_file(x)]
        #self.h_transform = test_h_transform()
        #self.s_transform = test_s_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        #return ToTensor()(h_image), ToTensor()(s_image)#image_name,
        return {'A': h_image, 'B': s_image}

    def __len__(self):
        return len(self.h_filenames)

class TestDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder1, self).__init__()
        # self.h_path = dataset_dir + '/hazy'  # '/hazy' indoor  '/low' #
        # self.s_path = dataset_dir + '/gt'  # '/clear' '/high' #
        # self.h_path = dataset_dir +'/outdoor' + '/hazy'  # '/hazy' indoor  '/low' #
        # self.s_path = dataset_dir +'/outdoor' + '/gt'  # '/clear' '/high' #
        # self.h_path = dataset_dir + '/rain'# + '/hazy'  # '/hazy' indoor  '/low' #
        # self.s_path = dataset_dir + '/norain'# + '/gt'  # '/clear' '/high' #
        # self.h_path = dataset_dir + '/rain'  # + '/hazy'  # '/hazy' indoor  '/low' #  input_crops
        # self.s_path = dataset_dir + '/norain'  # + '/gt'  # '/clear' '/high' #

        # self.h_path = dataset_dir + 'input_crops' #'/indoor' + '/hazy_new256'  # '/hazy' indoor
        # self.s_path = dataset_dir + 'target_crops' #'/indoor' + '/gt_new256'

        # self.h_path = dataset_dir + '/low' #+ '/hazy'  # '/hazy' indoor
        # self.s_path = dataset_dir + '/high' #+ '/gt'
        # self.h_path = dataset_dir + '/data'  # + '/hazy'  # '/hazy' indoor
        # self.s_path = dataset_dir + '/gt'  # + '/gt'
        # self.h_path = dataset_dir + '/synthetic'  # + '/hazy'  # '/hazy' indoor
        # self.s_path = dataset_dir + '/gt'  # + '/gt'
        #self.s_path = dataset_dir + '/t'#'/gt'
        # self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path))[0:10] if
        #                     is_image_file(x)]
        # self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path))[0:10] if
        #                     is_image_file(x)]
        self.h_path = dataset_dir + '/data'  # R1400 # + '/hazy'  # '/hazy' indoor  '/low' #  input_crops
        self.s_path = dataset_dir + '/gt'  # + '/gt'  # '/clear' '/high' #
        # self.s_path = dataset_dir + '/t'#'/gt'
        # self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path))[0:10] if
        #                     is_image_file(x)]
        # self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path))[0:10] if
        #                     is_image_file(x)]
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path))  if is_image_file(x)]  # [0:4808][0:9600]#p in range(10)
        self.s_filenames = [join(self.s_path,  x) for x in natsort.natsorted(listdir(self.s_path))  if  is_image_file(x)]#[0:4122]
        # self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        # self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path))  if  is_image_file(x)]  # for p in range(10)
        # self.hd_filenames = [join(self.hd_path, x) for x in natsort.natsorted(listdir(self.hd_path)) if  is_image_file(x)]  # for p in range(10)
        # self.h_transform = test_transform()
        self.s_transform = test_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        # print("Number of ground truth images:", len(self.s_filenames))
        # print("Number of synthetic images:", len(self.h_filenames))
        # print("Expected number of synthetic images per ground truth:", len(self.h_filenames) / len(self.s_filenames))

        h_image =  self.s_transform((Image.open(self.h_filenames[index])))
        # hd_image = self.s_transform((Image.open(self.hd_filenames[index])))
        s_image = self.s_transform( (Image.open(self.s_filenames[index])))
        # return {'A': s_image, 'B': h_image, 'Bd': hd_image}
        # return {'A': h_image, 'B': s_image}
        return h_image, s_image,image_name

    def __len__(self):
        return len(self.h_filenames)

class TestDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder2, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/data'  # '/hazy' simu_256  haze_new
        self.s_path = dataset_dir + '/gt'  # hazy_new_256 # '/gt' # haze_new clear_256 JPEGImages_new  haze_new
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if
                            is_image_file(x)]  # for p in range(10)
        #self.s_path = dataset_dir + '/t'#'/gt'
        # self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        # self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) for p in range(10) if is_image_file(x)]#
        self.h_transform = test_transform()
        #self.s_transform = test_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = self.h_transform((Image.open(self.h_filenames[index])))
        s_image = self.h_transform((Image.open(self.s_filenames[index]))) #ToTensor
        # return {'A': s_image, 'B': h_image}
        # return {'A': h_image, 'B': s_image}
        return h_image, s_image,image_name

    def __len__(self):
        return len(self.h_filenames)
#
# class TestDatasetFromFolder2(Dataset):
#     def __init__(self, dataset_dir):
#         super(TestDatasetFromFolder2, self).__init__()
#         #self.h_path = dataset_dir + '/h'#'/hazy'
#         self.h_path = dataset_dir + '/h'#'/hazy'
#         self.s_path = dataset_dir + '/t'#'/clear'
#         #self.s_path = dataset_dir + '/t'#'/gt'
#         self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
#         self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if is_image_file(x)]
#         #self.h_transform = test_h_transform()
#         #self.s_transform = test_s_transform()
#
#     def __getitem__(self, index):
#         image_name = self.h_filenames[index].split('/')[-1]
#         h_image = ToTensor()(Image.open(self.h_filenames[index]))
#         s_image = ToTensor()(Image.open(self.s_filenames[index]))
#         return {'A': h_image, 'B': s_image}
#
#     def __len__(self):
#         return len(self.h_filenames)
#


class TrainDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir_h, dataset_dir_s,dataset_dir_trans, crop_size):
        super(TrainDatasetFromFolder1, self).__init__()
        # self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:350] if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:520] for p in range(35) if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:10] for p in range(35) if is_image_file(x)]

        self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h)) if is_image_file(x)] #for p in range(11)
        self.image_filenames_trans = [join(dataset_dir_trans, x) for x in natsort.natsorted(listdir(dataset_dir_trans)) if is_image_file(x)]

        self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:2798] if is_image_file(x)] #for p  in range(10)

        crop_size = calculate_valid_crop_size(crop_size)
        self.h_transform = train_s_transform(crop_size)
        self.s_transform = train_s_transform(crop_size)
        self.trans_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        h_image = self.h_transform(Image.open(self.image_filenames_h[index]))
        trans_image = self.trans_transform(Image.open(self.image_filenames_trans[index]))
        s_image = self.s_transform(Image.open(self.image_filenames_s[index]))

        #return h_image, s_image

        return {'A': h_image, 'B': s_image,'T':trans_image}

    def __len__(self):
        return  len(self.image_filenames_h) #max(len(self.image_filenames_h), len(self.image_filenames_s))

class TrainDatasetFromFolder6(Dataset):
    def __init__(self, dataset_dir_c,dataset_dir_h,crop_size ):
        super(TrainDatasetFromFolder2, self).__init__()

        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))  for p in range(35)  if is_image_file(x)]#[0:4808][0:9600]#p in range(10)
        # self.image_filenames_B = [join(dataset_dir_h,  x) for (x) in natsort.natsorted(listdir(dataset_dir_h))  if  is_image_file(x)]#[0:4122]
        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))  if is_image_file(x)]  # for p in range(35)[0:4808][0:9600]#p in range(10)
        self.image_filenames_B = [join(dataset_dir_h,  x) for (x) in natsort.natsorted(listdir(dataset_dir_h)) if is_image_file(x)]#[0:4122]
        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))     for p in range(10)  if is_image_file(x)]  # [0:4808][0:9600]#p in range(10)
        # self.image_filenames_B = [join(dataset_dir_h, x) for (x) in natsort.natsorted(listdir(dataset_dir_h))  for p in range(10)   if  is_image_file(x)]  # [0:4122]
        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c)) if is_image_file(x)]  # [0:4808][0:9600]#p in range(10)
        # self.image_filenames_B = [join(dataset_dir_h, x) for (x) in natsort.natsorted(listdir(dataset_dir_h)) if   is_image_file(x)]  # [0:4122]
        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c)) for p in range(100) if is_image_file(x)]  # [0:4808][0:9600]#p in range(10)
        # self.image_filenames_B = [join(dataset_dir_h, x) for (x) in natsort.natsorted(listdir(dataset_dir_h)) for p in   range(100) if is_image_file(x)]
        self.image_filenames_A = self.image_filenames_A * 7
        self.image_filenames_B = self.image_filenames_B * 7
        crop_size = calculate_valid_crop_size(crop_size)
        self.c_transform = train_h_transform(crop_size)
        self.crop_size = calculate_valid_crop_size(crop_size) #crop_size

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        B_image = self.c_transform(Image.open(self.image_filenames_B[index]))
        w = A_image.size(2)
        h = A_image.size(1)
        w_offset = random.randint(0, max(0, w - self.crop_size - 1))
        h_offset = random.randint(0, max(0, h - self.crop_size - 1))
        #
        A_image = A_image[:, h_offset:h_offset + self.crop_size,
                  w_offset:w_offset + self.crop_size]
        B_image = B_image[:, h_offset:h_offset + self.crop_size,
                  w_offset:w_offset + self.crop_size]

        return  B_image,A_image#,R_image


    def __len__(self):
        return  len(self.image_filenames_A) #max(len(self.image_filenames_h), len(self.image_filenames_s))
    def augData(self,data,target):
        #if self.train:
        if 1:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        # data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target


class TrainDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir_c,dataset_dir_h,crop_size ):
        super(TrainDatasetFromFolder2, self).__init__()

        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))  for p in range(35)  if is_image_file(x)]#[0:4808][0:9600]#p in range(10)
        # self.image_filenames_B = [join(dataset_dir_h,  x) for (x) in natsort.natsorted(listdir(dataset_dir_h))  if  is_image_file(x)]#[0:4122]
        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c)) for p in range(50) if is_image_file(x)]  # for p in range(35)[0:4808][0:9600]#p in range(10)
        # self.image_filenames_B = [join(dataset_dir_h,  x) for (x) in natsort.natsorted(listdir(dataset_dir_h)) if is_image_file(x)]#[0:4122]
        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))     for p in range(10)  if is_image_file(x)]  # [0:4808][0:9600]#p in range(10)
        # self.image_filenames_B = [join(dataset_dir_h, x) for (x) in natsort.natsorted(listdir(dataset_dir_h))  for p in range(10)   if  is_image_file(x)]  # [0:4122]
        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c)) if is_image_file(x)]  # [0:4808][0:9600]#p in range(10)
        self.image_filenames_B = [join(dataset_dir_h, x) for (x) in natsort.natsorted(listdir(dataset_dir_h)) if   is_image_file(x)]  # [0:4122]
        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c)) for p in range(100) if is_image_file(x)]  # [0:4808][0:9600]#p in range(10)
        # self.image_filenames_B = [join(dataset_dir_h, x) for (x) in natsort.natsorted(listdir(dataset_dir_h)) for p in   range(100) if is_image_file(x)]
        # self.image_filenames_A = self.image_filenames_A * 3
        # self.image_filenames_B = self.image_filenames_B * 3
        crop_size = calculate_valid_crop_size(crop_size)
        self.c_transform = train_h_transform(crop_size)
        self.crop_size = calculate_valid_crop_size(crop_size) #crop_size

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        B_image = self.c_transform(Image.open(self.image_filenames_B[index]))
        w = A_image.size(2)
        h = A_image.size(1)
        w_offset = random.randint(0, max(0, w - self.crop_size - 1))
        h_offset = random.randint(0, max(0, h - self.crop_size - 1))
        #
        A_image = A_image[:, h_offset:h_offset + self.crop_size,
                  w_offset:w_offset + self.crop_size]
        B_image = B_image[:, h_offset:h_offset + self.crop_size,
                  w_offset:w_offset + self.crop_size]

        return  B_image,A_image#,R_image



    def __len__(self):
        return  len(self.image_filenames_A) #max(len(self.image_filenames_h), len(self.image_filenames_s))
    def augData(self,data,target):
        #if self.train:
        if 1:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        # data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target


import torchvision.transforms as tfs
from torchvision.transforms import functional as FF

class TrainDatasetFromFolder3(Dataset):
    def __init__(self, dataset_dir_c,dataset_dir_h,dataset_real,crop_size ):
        super(TrainDatasetFromFolder3, self).__init__()
        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c)) [0:2061]  for p in range(2)  if is_image_file(x)]
        # self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h)) [0:4122]  if is_image_file(x)]
        # self.image_filenames_C = [join(dataset_real, x) for x in natsort.natsorted(listdir(dataset_real)) [0:4122]   if is_image_file(x)]

        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))[0:1000] for p in range(3) if  is_image_file(x)]#[0:9600]
        self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:1000]  for p in range(3)  if      is_image_file(x)]
        self.image_filenames_C = [join(dataset_real, x) for x in natsort.natsorted(listdir(dataset_real))[0:1500] for p in range(2)  if is_image_file(x)]#for p  in range(2)

        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))   if
        #                           is_image_file(x)]
        # self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))   if
        #                           is_image_file(x)]
        # self.image_filenames_C = [join(dataset_real, x) for x in natsort.natsorted(listdir(dataset_real))[0:4800]  for p in range(2) if is_image_file(x)]
        # self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))[0:5000]   if is_image_file(x)]#[0:9600]for p in range(2)
        # self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:1000]  for p in range(5) if  is_image_file(x)]#[0:4122]
        # self.image_filenames_C = [join(dataset_real, x) for x in natsort.natsorted(listdir(dataset_real))[0:2500] for p in range(2)  if is_image_file(x)]# for p in range(5)

       # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:5000] if is_image_file(x)] #for p  in range(10)

        crop_size = calculate_valid_crop_size(crop_size)
        self.c_transform = train_trans_transform(crop_size)
        self.h_transform = train_trans_transform(crop_size)
        self.r_transform = train_trans_transform(crop_size)

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        B_image = self.h_transform(Image.open(self.image_filenames_B[index]))
        # R_image = self.r_transform(Image.open(self.image_filenames_C[index]))

        return B_image, A_image

        # return {'A': A_image, 'B':B_image,'R':R_image}#'B': s_image,

    def __len__(self):
        return  len(self.image_filenames_A) #max(len(self.image_filenames_h), len(self.image_filenames_s))


class TrainDatasetFromFolder4(Dataset):
    def __init__(self, dataset_dir_c, dataset_dir_r,dataset_real, crop_size):
        super(TrainDatasetFromFolder4, self).__init__()

        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c)) for p in range(3) if is_image_file( x)]  # for p in range(2) 8707 #for p in range(11)  # for p in range(10)#5687    27327   2000  #is_image_file(10*x)] [0:5687]
        self.image_filenames_C = [join(dataset_dir_r, x) for x in natsort.natsorted(listdir(dataset_dir_r)) if  is_image_file(x)]  # for p in range(3)
        self.image_filenames_R = [join(dataset_real, x) for x in natsort.natsorted(listdir(dataset_real)) if  is_image_file(x)]  # [0:4122]

        crop_size = calculate_valid_crop_size(crop_size)
        self.r_transform = train_trans_transform(crop_size)
        # self.h_transform = train_s_transform(crop_size)
        self.c_transform = train_trans_transform(crop_size)

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        # B_image = self.h_transform(Image.open(self.image_filenames_B[index]))
        B_image = self.c_transform(Image.open(self.image_filenames_C[index]))
        #1203
        # RR_image = self.r_transform(Image.open(self.image_filenames_R[index]))
        #return h_image, s_image

        return B_image,A_image#,R_image#{'A': A_image, 'B':R_image, 'R':RR_image}#'B': s_image,

        # return {'A': A_image, 'B': B_image}

    def __len__(self):
        return  len(self.image_filenames_A) #max(len(self.image_filenames_h), len(self.image_filenames_s))
class TrainDatasetFromFolder5(Dataset):
    def __init__(self, dataset_dir_c, dataset_dir_h, crop_size):
        super(TrainDatasetFromFolder5, self).__init__()

        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))[0:2061] for p in range(11) if is_image_file(x)]   # for p in range(10)#5687    27327   2000  #is_image_file(10*x)] [0:5687]
        self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:22671] if is_image_file(x)]  #[0:5687]  54600 22671 14427

        crop_size = calculate_valid_crop_size(crop_size)
        self.c_transform = train_s_transform(crop_size)
        # self.h_transform = train_s_transform(crop_size)
        # self.r_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        B_image = self.c_transform(Image.open(self.image_filenames_B[index]))
        # A_image =Image.open(self.image_filenames_A[index])
        # B_image =Image.open(self.image_filenames_B[index])
        # image = self.c_transform(cat([A_image,B_image],0))
        # A_image= image[0:2,:,:]# B_image = self.c_transform(Image.open(self.image_filenames_B[index]))
        # B_image= image[3:5,:,:]
        # R_image = self.r_transform(Image.open(self.image_filenames_C[index]))

        #return h_image, s_image

        return {'A': A_image,'B': B_image }#'B': s_image,

        # return {'A': A_image, 'B': B_image}

    def __len__(self):
        return  len(self.image_filenames_A) #max(len(self.image_filenames_h), len(self.image_filenames_s))

