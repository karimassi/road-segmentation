import torch
import torchvision.io as io
import os
import matplotlib.image as mpimg
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch


# TODO: load to device
def load_mask(path):
    gt = torch.tensor(mpimg.imread(path))
    road = (gt > foreground_threshold) * 1.0
    background = (gt < foreground_threshold) * 1.0
    mask = torch.zeros([2, gt.shape[0], gt.shape[1]], dtype=torch.float32)

    mask[0] = road
    mask[1] = background

    return mask

class ImageMaskDataset(Dataset):
    
    def __init__(self, img_dir, gt_dir, angle):
        super().__init__()

        self.img_dir = img_dir
        self.gt_dir = gt_dir

        pil_to_tensor = transforms.ToTensor()

        #load data
        self.files = os.listdir(img_dir)
        self.n = len(self.files)
        
        #self.images = [io.read_image(img_dir + self.files[i]) for i in range(self.n)]
        self.images = [TF.rotate(pil_to_tensor(Image.open(img_dir + self.files[i]).convert('RGB')), angle) for i in range(self.n)]
        #self.masks = [torch.tensor(mpimg.imread(gt_dir + self.files[i])).unsqueeze(dim=0) for i in range(self.n)]
        #self.masks = [mpimg.imread(gt_dir + self.files[i]) for i in range(self.n)]
        self.masks = [TF.rotate(load_mask(gt_dir + self.files[i]), angle) for i in range(self.n)]
      

    def __len__(self): 
        return self.n
        
    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

class FullSubmissionImageDataset(Dataset):
    
    def __init__(self, test_dir):
        super().__init__()

        self.test_dir = test_dir

        transform = transforms.ToTensor()

        #load data
        files = os.listdir(test_dir)
        self.images = [(int(f[5:]), transform(Image.open(test_dir + f + "/" + f + ".png").convert('RGB'))) for f in files]
      
    def __len__(self): 
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx]

