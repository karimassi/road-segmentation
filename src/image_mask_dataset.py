import torch
import torchvision.io as io
import os
import matplotlib.image as mpimg
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# TODO: load to device
def load_mask(path):
  gt = torch.tensor(mpimg.imread(path))
  first_channel = (gt > 0.5) * 1.0
  second_channel = (gt < 0.5) * 1.0
  mask = torch.zeros([2, gt.shape[0], gt.shape[1]], dtype=torch.float32)
  
  mask[0] = first_channel
  mask[1] = second_channel
  
  return mask

class ImageMaskDataset(Dataset):
    
    def __init__(self, img_dir, gt_dir, transform = None):
      super().__init__()

      self.img_dir = img_dir
      self.gt_dir = gt_dir
        
      #if transform is None:
      #  transform = transforms.Compose([])

      transform = transforms.Compose([
          transforms.ToTensor()
      ])

      #load data
      self.files = os.listdir(img_dir)
      self.n = len(self.files)
      
      #self.images = [io.read_image(img_dir + self.files[i]) for i in range(self.n)]
      self.images = [transform(Image.open(img_dir + self.files[i]).convert('RGB')) for i in range(self.n)]
      #self.masks = [torch.tensor(mpimg.imread(gt_dir + self.files[i])).unsqueeze(dim=0) for i in range(self.n)]
      #self.masks = [mpimg.imread(gt_dir + self.files[i]) for i in range(self.n)]
      self.masks = [load_mask(gt_dir + self.files[i]) for i in range(self.n)]
      

    def __len__(self): 
        return self.n
        
    def __getitem__(self, idx):
      return self.images[idx], self.masks[idx]

