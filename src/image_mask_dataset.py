import torch
import os
import numpy as np
import torchvision.io as io
import matplotlib.image as mpimg
from torch.utils.data import Dataset
from skimage.util.shape import view_as_windows

class ImageMaskDataset(Dataset):

    @staticmethod
    def convert_to_one_hot(gt_img):
        # converts groundtruth image to one-hot encoding
        gt = torch.tensor(gt_img)
        road = (gt > 0.5) * 1.0
        background = (gt < 0.5) * 1.0
        mask = torch.zeros([2, gt.shape[0], gt.shape[1]], dtype=torch.float32)

        mask[0] = road
        mask[1] = background

        return mask

    @staticmethod
    def extract_patches(sat_img, gt_img):
        # extracts patches of size 224x224 with stride 64

        sat_img = np.transpose(sat_img, axes=(2, 0, 1))

        sat_windows = view_as_windows(sat_img, window_shape=(3, 224, 224), step=64)
        gt_windows = view_as_windows(gt_img, window_shape=(224, 224), step=64)

        sat_patches = []
        gt_patches = []
        for i in range(sat_windows.shape[1]):
            for j in range(sat_windows.shape[2]):
                sat_patch = np.transpose(sat_windows[0][i][j], axes=(1, 2, 0))
                gt_patch = gt_windows[i][j]
                sat_patches.append(sat_patch)
                gt_patches.append(gt_patch)

        return sat_patches, gt_patches
    
    def __init__(self, img_dir, gt_dir, transform=None):
        super().__init__()

        self.img_dir = img_dir
        self.gt_dir = gt_dir

        # load data
        self.files = os.listdir(img_dir)
        self.n = len(self.files)

        self.images = []
        self.masks = []
        for i in range(self.n):
            sat_img = mpimg.imread(img_dir + self.files[i])
            gt_img = mpimg.imread(gt_dir + self.files[i])
            img_patches, gt_patches = self.extract_patches(sat_img, gt_img)
            self.images += [torch.from_numpy(img).permute(2, 0, 1) for img in img_patches]
            self.masks += [self.convert_to_one_hot(gt) for gt in gt_patches]

        # apply transformation
        if transform is not None:
            self.images = [transform(img) for img in self.images]
            self.masks = [transform(mask) for mask in self.masks]

    def __len__(self): 
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class FullSubmissionImageDataset(Dataset):
    
    def __init__(self, test_dir):
        super().__init__()

        self.test_dir = test_dir

        #load data
        files = os.listdir(test_dir)
        self.images = [(int(f[5:]), torch.tensor(mpimg.imread(test_dir + f + "/" + f + ".png")).permute(2, 0, 1)) for f in files]
      
    def __len__(self): 
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx]

