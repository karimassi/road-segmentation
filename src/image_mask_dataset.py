import torch
import torchvision.io as io
import os
import matplotlib.image as mpimg
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# TODO: vectorize this
# TODO: load to device
def load_mask(path):
  gt = mpimg.imread(path)
  mask = torch.zeros([2, gt.shape[0], gt.shape[1]], dtype=torch.float32)
  for i in range(gt.shape[0]):
    for j in range(gt.shape[1]):
      if(gt[i][j] > 0.5):
        # first channel is for foreground
        # second channel is for background
        mask[0][i][j] = 1.0
      else:
        mask[1][i][j] = 1.0

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

