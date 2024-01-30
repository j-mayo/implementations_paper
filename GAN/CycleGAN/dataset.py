from torch.utils.data import Dataset
from torch import randint
from torchvision.transforms import transforms

from PIL import Image
import os

class CycleGANCustomDataset(Dataset):
    def __init__(self, dataset_path, mode='train', res=128, serial_batches=False):
        super(CycleGANCustomDataset, self).__init__()
        self.path_A = dataset_path + "/" + mode + "A/"
        self.path_B = dataset_path + "/" + mode + "B/"
        
        self.img_list_A = sorted(os.listdir(self.path_A))
        self.img_list_B = sorted(os.listdir(self.path_B))
        
        self.len_A = len(self.img_list_A)
        self.len_B = len(self.img_list_B)
        self.res = res
        self.serial_batches = serial_batches
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                          transforms.Resize(res, transforms.InterpolationMode.BICUBIC),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    def __len__(self):
        return max(self.len_A, self.len_B)
    
    
    def __getitem__(self, index):
        img_A = self.transform(Image.open(self.path_A+self.img_list_A[index % self.len_A]).convert('RGB'))
        if self.serial_batches:
            index_b = index % self.len_B
        else:
            index_b = int(randint(0, self.len_B - 1, (1,))[0])
        img_B = self.transform(Image.open(self.path_B+self.img_list_B[index_b]).convert('RGB'))
        #print(img_A, img_B)
        return {'A': img_A, 'B': img_B}