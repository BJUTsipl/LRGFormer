#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from thop import profile,clever_format
# from unet import UNet
from  CS_unet1027 import UNet_emb#,UNet_embDIS #UNet_emb #UNet
from  utilss import *
import torchvision
import os
import json
import torchvision.utils as vutils
from torchvision.models import vgg16
from perceptual import LossNetwork
from ECLoss import *


from fnmatch import fnmatch

vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
for param in vgg_model.parameters():
     param.requires_grad = False

loss_network = LossNetwork(vgg_model).cuda()
loss_network.eval()
# ckpts_dir = '../ckpts/outdoor-g4bi3_mlp/dehamer-epoch23-0.09184.pt'
torch.backends.cudnn.enabled = False
class dehamer(object):
    """Implementation of dehamer from Guo et al. (2022)."""

    def __init__(self, params, trainable):
        """Initializes model."""
        self.p = params
        self.trainable = trainable
        self._compile()


    def _compile(self):

        # self.model = UNet_emb()
        self.model = UNet_emb()
        print("UNet_emb have {} parameters in total".format(sum(x.numel() for x in  self.model.parameters())))

        # Print model parameters and FLOPs


        # self.modelmeta = UNet_emb()
        # self.encoder = UNet_embDIS.encoder()

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])


            # self.optim2 = Adam(self.model.parameters(),
            #                   lr=self.p.learning_rate*0.5,
            #                   betas=self.p.adam[:2],
            #                   eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/10, factor=0.5, verbose=True)

            # self.scheduler =lr_scheduler.CosineAnnealingLR(self.optim, T_max=100)#eta_min=0.000001,
            # print('Learning rate sets to {}.'.format( self.optim.parameters['lr']))
            # self.scheduler2 = lr_scheduler.ReduceLROnPlateau(self.optim2,
            #                                                 patience=self.p.nb_epochs / 4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        # if self.use_cuda:
        self.model = self.model.cuda()
        # self.modelmeta = self.modelmeta.cuda()
        # self.model.load_state_dict(torch.load('./dehamer-epoch53-0.02885.pt'), strict=False)
        if self.trainable:
                self.loss = self.loss.cuda()
                # self.loss = self.loss.cuda()
        # self.model = torch.nn.DataParallel(self.model)


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = self.p.dataset_name #f'{datetime.now():{self.p.dataset_name}-%m%d-%H%M}'
            if self.p.ckpt_overwrite:
                ckpt_dir_name = self.p.dataset_name

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/dehamer-{}.pt'.format(self.ckpt_dir, self.p.dataset_name)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/dehamer-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1 , valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/dehamer-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        # if self.use_cuda:
        self.model.load_state_dict(torch.load(ckpt_fname))
        # else:
        #     self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""
        # import pdb;pdb.set_trace()
        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)
        # self.scheduler2.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = self.p.loss.upper()#f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('./results'):
        os.makedirs('./results/source')
        os.makedirs('./results/target')
        os.makedirs('./results/dehaze')
    def eval(self, valid_loader):
        with torch.no_grad():
            self.model.train(False)

            valid_start = datetime.now()
            loss_meter = AvgMeter()
            psnr_meter = AvgMeter()

            for batch_idx, (source, target,haze_name) in enumerate(valid_loader):
                # if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

                h, w = source.size(2), source.size(3)
                haze_name= haze_name[0]

                pad_h = h % 16
                pad_w = w % 16
                source = source[:, :, 0:h - pad_h, 0:w - pad_w]
                target = target[:, :, 0:h - pad_h, 0:w - pad_w]

                # dehaze
                # source_dehazed = self.model(source)
                source_dehazed = self.model.Encoder(source)
                # source_dehazed  = self.model.Decoder(cor, trr )
                vutils.save_image(source.data, './results/source/'+haze_name, padding=0,
                                  normalize=True)  # False
                vutils.save_image(target.data, './results/target/'+haze_name , padding=0, normalize=True)
                vutils.save_image(source_dehazed.data, './results/dehaze/' +haze_name, padding=0,
                                  normalize=True)
                # Update loss
                loss = self.loss(source_dehazed, target)
                loss_meter.update(loss.item())

                # Compute PSRN
                for i in range(source_dehazed.shape[0]):
                    # import pdb;pdb.set_trace()
                    source_dehazed = source_dehazed.cpu()
                    target = target.cpu()
                    psnr_meter.update(psnr(source_dehazed[i], target[i]).item())



            valid_loss = loss_meter.avg
            valid_time = time_elapsed_since(valid_start)[0] 
            psnr_avg = psnr_meter.avg

            return valid_loss, valid_time, psnr_avg 
 

    def train(self, train_loader, valid_loader): 
        """Trains denoiser on training set."""  
 
        self.model.train(True)
        
        if self.p.ckpt_load_path is not None:
            self.model.load_state_dict(torch.load(self.p.ckpt_load_path), strict=False)  ##zhushi
            print('The pretrain model is loaded.')
        self._print_params()
        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'dataset_name': self.p.dataset_name, 
                 'train_loss': [],
                 'valid_loss': [], 
                 'valid_psnr': []}  
  
        # Main training loop 
        train_start = datetime.now()
        ite=0
        # lr=[]
        for epoch in range(self.p.nb_epochs): 
            print('EPOCH {:d} / {:d}'.format(epoch + 1 , self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()
            # if epoch>9 and epoch <20:
            #     self.optim.param_groups[0]['lr']*=0.5
            # if epoch>19 and epoch <50:
            #     self.optim.param_groups[0]['lr']*=0.1
            # if epoch>49 and epoch <100:
            #     self.optim.param_groups[0]['lr']*=0.01


            print('learningrate sets to {}'.format(self.optim.param_groups[0]['lr']))
            # Minibatch SGD
            for batch_idx, (source, target ) in enumerate(train_loader):
              batch_start = datetime.now()
              progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                # if self.use_cuda:
              source = source.cuda()
              target = target.cuda()
              if source.size(1) == 3 and target.size(1) == 3:
                source_dehazed = self.model.Encoder(source)
                self.optim.zero_grad()
                loss_de =  self.loss(source_dehazed , target)  + loss_network(source_dehazed , target) * 0.04\

                loss = 100*loss_de# + 0.01 *loss_cf + 0.01 *loss_dcp

                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                # self.model.load_state_dict(tmp_weight)
                # self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                # lr.append(self.optim.state_dict()['param_groups'][0]['lr'])
                # lr.append(self.scheduler.get_lr()[0])
                #


                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg , loss_de ,  time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()
                ite += 1
                if ite % 100 == 0:
                    # print(output)
                    vutils.save_image(source.data, './source.png', normalize=True)
                    vutils.save_image(source_dehazed.data, './source_dehazed.png', normalize=True)
                    vutils.save_image(target.data, './target.png', normalize=True)
                    # vutils.save_image(target_dehazed.data, './target_dehaze.png', normalize=True)
                    # vutils.save_image(source_dehazed .data, './source_dehazed1.png', normalize=True)
                    # vutils.save_image(fake_target.data, './fake_target.png', normalize=True)
                    # vutils.save_image(fake_source.data, './fake_source.png', normalize=True)
                    # vutils.save_image(real.data, './real.png', normalize=True)
                    # vutils.save_image(real_dehazed.data, './real_dehazed.png', normalize=True)

            # Epoch end, save and reset tracker
            # print('learningrate sets to {}'.format(self.optim.param_groups[0]['lr']))
            # self.scheduler.step()
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()
        # plt.plot(np.arange(len(lr)),lr)
        # plt.savefig('lr.jpg')


        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))




