import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import CycleGANCustomDataset
from module import Generator, Discriminator
from model import CycleGAN
from scheduler import get_scheduler

def train(model, dataloader, num_epochs, lr_args):
        # log 관련
        log_dict = {'loss_G': [], 'loss_D': []}
        cnt = 0
        
        model.set_optimizer(lr_args['lr'], 0.5)
        scheduler_D = get_scheduler(model.opt_D, num_epochs, lr_args)
        scheduler_G = get_scheduler(model.opt_G, num_epochs, lr_args)
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader):
                # TODO : dataloader에 collate 작성해서 A, B를 따로 빼 올 수 있도록 하기
                #print(i, data)
                x_batch_A = data['A'].to(model.device)
                x_batch_B = data['B'].to(model.device)

                # G
                model.opt_G.zero_grad()
                model.opt_D.zero_grad()
                for param in model.D_A.parameters():
                    param.requires_grad=False
                for param in model.D_B.parameters():
                    param.requires_grad=False

                G_A_output = model.G_A(x_batch_A) # fake B
                D_B_G_A_output = model.D_B(G_A_output) # discriminate fake B
                B_G_A_output = model.G_B(G_A_output) # reconstructed A
                G_B_output = model.G_B(x_batch_B) # fake A
                D_A_G_B_output = model.D_A(G_B_output) # discriminate fake A
                A_G_B_output = model.G_A(G_B_output) # reconstructed B
                
                real_y = torch.ones_like(D_B_G_A_output).to(model.device)
                fake_y = torch.ones_like(D_B_G_A_output).to(model.device)
                
                g1 = model.loss_GAN(D_B_G_A_output, real_y.detach())
                g2 = model.loss_GAN(D_A_G_B_output, real_y.detach())
                loss_GAN = g1 + g2
                loss_cycle = model.lambda_cyc * model.loss_cycle(B_G_A_output, x_batch_A, A_G_B_output, x_batch_B)
                loss_id = model.lambda_id * model.loss_id(G_A_output, x_batch_A, G_B_output, x_batch_B)
                loss_G = loss_GAN + loss_cycle + loss_id
                
                loss_G.backward()
                model.opt_G.step()
                
                # D
                for param in model.D_A.parameters():
                    param.requires_grad=True
                for param in model.D_B.parameters():
                    param.requires_grad=True
                
                
                loss_GAN_D_A = (model.loss_GAN(model.D_A(x_batch_A), real_y) + model.loss_GAN(model.D_A(G_B_output.detach()), fake_y)) / 2
                loss_GAN_D_A.backward()
                loss_GAN_D_B = (model.loss_GAN(model.D_B(x_batch_B), real_y) + model.loss_GAN(model.D_B(G_A_output.detach()), fake_y)) / 2 # grad true인 상태에서!
                loss_GAN_D_B.backward()
                
                model.opt_D.step()
                
                
                # TODO - logging
                
                if cnt % 100 == 0:
                    log_dict['loss_G'].append(loss_G.item())
                    log_dict['loss_D'].append(loss_GAN_D_A.item() + loss_GAN_D_B.item())
                    print(f'epoch {epoch}, loss_GAN {g1.item()} {g2.item()}, loss_cycle {loss_cycle.item()}, loss_id {loss_id.item()}, loss_D {loss_GAN_D_A.item() + loss_GAN_D_B.item()}')
                
            scheduler_D.step()
            scheduler_G.step()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CycleGAN(Generator, Discriminator, 64, 64, 3, (3, 256, 256), lambda_cyc=10, lambda_id=0.5, gan_loss_type=nn.MSELoss, device=device)

    train_dataset = CycleGANCustomDataset("../data/maps", 'train', 256, False)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    
    lr_args = {'lr': 0.0002, 'lr_policy': 'linear', 'lr_decay_iters': 50, 'epoch_count': 1, 'n_epochs': 100, 'n_epochs_decay': 100} # 100 epoch동안 그대로, 100 epoch 동안 decay
    model.train(train_dataloader, 201, lr_args)
    
    # TODO : remove hard coding
    torch.save(model.G_B.state_dict(), "models/cyclegan_generator_A.pt")
    torch.save(model.D_B.state_dict(), "models/cyclegan_discriminator_A.pt")
    torch.save(model.G_B.state_dict(), "models/cyclegan_generator_B.pt")
    torch.save(model.D_B.state_dict(), "models/cyclegan_discriminator_B.pt")