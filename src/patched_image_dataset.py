import torch
import os, sys
import torchvision.io as io
import matplotlib.image as mpimg
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class PatchedSatImagesDataset(Dataset):
    img_size = (400, 400)
    patch_size = (16, 16)
    
    def __init__(self, training_img_path, training_gt_path, foreground_threshold = None, transform = None):
        """
            Dataset for the traing data, this dataset is already patched

            @param training_img_path    : (string)             path to the training sat images
            @param training_gt_path     : (string)             path to the groundtruth images
            @param foreground_threshold : (float, optional)     if a value is provided then the label is 1 if the mean of the patch is greater than this value. 
                                                               if no value is provided, the mean is returned as label
            @param transform            : (callable, optional) a transformation to apply to each patch before returning it
        """
        super().__init__()
        
        if transform is None:
            transform = transforms.Compose([])
        
        # Reshape the ground truth images to be able to transform them 
        self.files = [{"sat" : transform(io.read_image(training_img_path + f)), 
                       "gt" : transform(torch.tensor(mpimg.imread(training_gt_path + f)).view(1, self.img_size[0], self.img_size[1]))} 
                      for f in sorted(os.listdir(training_img_path))]
        
        self.foreground_threshold = foreground_threshold
        
    def patch_per_img(self):
        return (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
    
    def __len__(self): 
        return len(self.files) * self.patch_per_img()
        
    def __getitem__(self, idx):
        files_number = idx // self.patch_per_img()
        patch_number = idx % self.patch_per_img()
        files = self.files[files_number]
        sat_img = files["sat"]
        gt_img = files["gt"]
        row_number = self.patch_size[0] * (patch_number // (self.img_size[0] // self.patch_size[0]))
        col_number = self.patch_size[1] * (patch_number % (self.img_size[0] // self.patch_size[0]))
        
        X = sat_img[:, row_number : row_number + self.patch_size[0], col_number : col_number + self.patch_size[1]] / 255
        Y = torch.mean(gt_img[:, row_number : row_number + self.patch_size[0], col_number : col_number + self.patch_size[1]])
         
        if self.foreground_threshold is not None:
            if Y > self.foreground_threshold :
                Y = torch.tensor(1.0)
            else :
                Y = torch.tensor(0.0)
        
        return X, Y

class PatchedTestSatImagesDataset(Dataset):
    img_size = (608, 608)
    patch_size = (16, 16)
    
    def __init__(self, test_img_path, transform = None):
        """
            Dataset for the testing data, this dataset is already patched

            @param test_img_path    : (string)             path to the testing sat images
            @param transform        : (callable, optional) a transformation to apply to each patch before returning it
        """
        super().__init__()
        
        if transform is None:
            transform = transforms.Compose([])
        
        self.files = [(int(f[5:]), transform(io.read_image(test_img_path + f + "/" + f + ".png"))) 
                      for f in sorted(os.listdir(test_img_path))]
    
    def patch_per_img(self):
        return (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
    
    def __len__(self): 
        return len(self.files) * self.patch_per_img()
        
    def __getitem__(self, idx):
        files_number = idx // self.patch_per_img()
        patch_number = idx % self.patch_per_img()
        img_id, sat_img = self.files[files_number]
        row_number = self.patch_size[0] * (patch_number // (self.img_size[0] // self.patch_size[0]))
        col_number = self.patch_size[1] * (patch_number % (self.img_size[0] // self.patch_size[0]))
        
        X = sat_img[:, row_number : row_number + self.patch_size[0], col_number : col_number + self.patch_size[1]] / 255
        
        return "{:03d}_{}_{}".format(img_id, col_number, row_number), X


