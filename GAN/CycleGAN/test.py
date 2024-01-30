import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import CycleGANCustomDataset
from module import Generator, Discriminator
from model import CycleGAN
from torchvision.transforms import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CycleGAN(Generator, Discriminator, 64, 64, 3, (3, 256, 256), lambda_cyc=10, lambda_id=0.5, gan_loss_type=nn.MSELoss, device=device)
transform = transforms.Compose([
                            transforms.ToTensor(),
                          transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
# TODO : remove hard coding
model.G_A.load_state_dict(torch.load("../models/cyclegan_generator_A.pt"))
#model.load_state_dict(torch.load("models/cyclegan_discriminator_A.pt"))
model.G_B.load_state_dict(torch.load("../models/cyclegan_generator_B.pt"))
#model.load_state_dict(torch.load("models/cyclegan_discriminator_B.pt"))

A_path = '../../data/maps/testA/5_A.jpg'
B_path = '../../data/maps/testB/5_B.jpg'

A_img = transform(Image.open(A_path).convert('RGB')).to(device)
B_img = transform(Image.open(B_path).convert('RGB')).to(device)

generated_B = to_pil_image(0.5 * model.G_A(A_img) + 0.5)
generated_A = to_pil_image(0.5 * model.G_B(B_img) + 0.5)

generated_A.save("generated_A.jpg")
generated_B.save("generated_B.jpg")
