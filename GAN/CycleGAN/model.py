import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch import optim
import itertools


class CycleGAN():
    
    def __init__(self, generator, discriminator, ngf, ndf, nc, img_size, device, lambda_cyc=10, lambda_id=0.5, gan_loss_type=nn.MSELoss):
        
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.img_size = img_size
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id
        self.G_A = generator(ngf, nc, img_size, use_dropout = 0.5).to(device)
        self.G_B = generator(ngf, nc, img_size, use_dropout = 0.5).to(device)
        self.D_A = discriminator(ndf, nc, img_size).to(device)
        self.D_B = discriminator(ndf, nc, img_size).to(device)
        self.gan_loss_type = gan_loss_type
        
        self.initialize_network(init_type='normal', init_gain=0.02)
    
    def loss_GAN(self, prediction, target):
        return self.gan_loss_type()(prediction, target)
    
    def loss_cycle(self, f_g_x, x, g_f_y, y):
        return nn.L1Loss()(f_g_x, x) + nn.L1Loss()(g_f_y, y)
    
    def loss_id(self, g_y, y, f_x, x):
        return nn.L1Loss()(g_y, y) + nn.L1Loss()(f_x, x)
    
    def set_optimizer(self, lr, beta1):
        self.opt_G = optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=lr, betas=(beta1, 0.999))
        self.opt_D = optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=lr, betas=(beta1, 0.999))
    
    
    def get_scheduler(self, optimizer, num_epochs, lr_args):
        lr_policy = lr_args['lr_policy']
        if lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + lr_args['epoch_count'] - num_epochs) / float(lr_args['n_epochs_decay'] + 1) # start epoch는 1로?
                return lr_l
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_args['lr_decay_iters'], gamma=0.1)
        elif lr_policy == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif lr_policy == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', lr_args['lr_policy'])

        return scheduler
    
    def initialize_network(self, init_type='normal', init_gain=0.02):
        
        def initialize_model(model):
            classname = model.__class__.__name__
            
            if hasattr(model, 'weight') and classname.find('Conv') != -1 or classname.find('Linear') != -1:
                if init_type == 'normal':
                    nn.init.normal_(model.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(model.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(model.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(model, 'bias') and model.bias is not None:
                    nn.init.constant_(model.bias.data, 0)
            # batchnorm
            elif classname.find('BatchNorm') != -1: # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                nn.init.normal_(model.weight.data, 1.0, gain=init_gain)
                nn.init.constant_(model.bias.data, 0)
        
        self.D_A.apply(initialize_model)
        self.D_B.apply(initialize_model)
        self.G_A.apply(initialize_model)
        self.G_B.apply(initialize_model)
    
    
    def train(self, dataloader, num_epochs, lr_args):
        # log 관련
        log_dict = {'loss_G': [], 'loss_D': []}
        cnt = 0
        
        self.set_optimizer(lr_args['lr'], 0.5)
        scheduler_D = self.get_scheduler(self.opt_D, num_epochs, lr_args)
        scheduler_G = self.get_scheduler(self.opt_G, num_epochs, lr_args)
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader):
                # TODO : dataloader에 collate 작성해서 A, B를 따로 빼 올 수 있도록 하기
                #print(i, data)
                x_batch_A = data['A'].to(self.device)
                x_batch_B = data['B'].to(self.device)

                # G
                self.opt_G.zero_grad()
                for param in self.D_A.parameters():
                    param.requires_grad=False
                for param in self.D_B.parameters():
                    param.requires_grad=False

                G_A_output = self.G_A(x_batch_A) # fake B
                D_B_G_A_output = self.D_B(G_A_output) # discriminate fake B
                B_G_A_output = self.G_B(G_A_output) # reconstructed A
                G_B_output = self.G_B(x_batch_B) # fake A
                D_A_G_B_output = self.D_A(G_B_output) # discriminate fake A
                A_G_B_output = self.G_A(G_B_output) # reconstructed B
                
                real_y = torch.ones_like(D_B_G_A_output).to(self.device)
                fake_y = torch.ones_like(D_B_G_A_output).to(self.device)
                
                g1 = self.loss_GAN(D_B_G_A_output, real_y.detach())
                g2 = self.loss_GAN(D_A_G_B_output, real_y.detach())
                loss_GAN = g1 + g2
                loss_cycle = self.lambda_cyc * self.loss_cycle(B_G_A_output, x_batch_A, A_G_B_output, x_batch_B)
                loss_id = self.lambda_id * self.loss_id(G_A_output, x_batch_A, G_B_output, x_batch_B)
                loss_G = loss_GAN + loss_cycle + loss_id
                
                loss_G.backward()
                self.opt_G.step()
                
                # D
                for param in self.D_A.parameters():
                    param.requires_grad=True
                for param in self.D_B.parameters():
                    param.requires_grad=True
                self.opt_D.zero_grad()
                
                loss_GAN_D_A = (self.loss_GAN(self.D_A(x_batch_A), real_y) + self.loss_GAN(self.D_A(G_B_output.detach()), fake_y)) / 2
                loss_GAN_D_A.backward()
                loss_GAN_D_B = (self.loss_GAN(self.D_B(x_batch_B), real_y) + self.loss_GAN(self.D_B(G_A_output.detach()), fake_y)) / 2 # grad true인 상태에서!
                loss_GAN_D_B.backward()
                
                self.opt_D.step()
                
                
                # TODO - logging
                
                if cnt % 100 == 0:
                    log_dict['loss_G'].append(loss_G.item())
                    log_dict['loss_D'].append(loss_GAN_D_A.item() + loss_GAN_D_B.item())
                    print(f'epoch {epoch}, loss_GAN {g1.item()} {g2.item()}, loss_cycle {loss_cycle.item()}, loss_id {loss_id.item()}, loss_D {loss_GAN_D_A.item() + loss_GAN_D_B.item()}')
                
            scheduler_D.step()
            scheduler_G.step()

        
                    
                    
                    
                    
                    