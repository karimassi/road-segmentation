import torch
from torch.utils.data import Dataset
import torchvision.io as io
import matplotlib.image as mpimg
import os, sys

root_dir = "/content/drive/Shareddrives/road-segmentation/data/" if "google.colab" in sys.modules else "data/"
img_path = root_dir + "training/images/"
gt_path = root_dir + "training/groundtruth/"
test_path = "test_set_images/"

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
        
        self.files = [{"sat" : io.read_image(training_img_path + f), "gt" : torch.tensor(mpimg.imread(training_gt_path + f))} for f in sorted(os.listdir(training_img_path))]
        self.foreground_threshold = foreground_threshold
        self.transform = transform
    
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
        row_number = patch_number // (self.img_size[0] // self.patch_size[0])
        col_number = patch_number % (self.img_size[0] // self.patch_size[0])
        
        X = sat_img[:, row_number : row_number + self.patch_size[0], col_number : col_number + self.patch_size[1]] / 255
        Y = gt_img[row_number : row_number + self.patch_size[0], col_number : col_number + self.patch_size[1]]
        
        if self.transform is not None:
            X = self.transform(X)
            Y = self.transform(Y)
            
        Y = torch.mean(Y)
 
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
        
        self.files = [io.read_image(test_img_path + f + "/" + f + ".png") for f in sorted(os.listdir(test_img_path))]
        self.transform = self.transform
    
    def patch_per_img(self):
        return (self.img_size[0] // self.patch_size[0]) * (self.mg_size[1] // self.patch_size[1])
    
    def __len__(): 
        return len(self.files) * self.patch_per_img()
        
    def __getitem__(self, idx):
        files_number = idx // self.patch_per_img()
        patch_number = idx % self.patch_per_img()
        sat_img = self.files[files_number]
        row_number = patch_number // (self.img_size[0] // self.patch_size[0])
        col_number = patch_number % (self.img_size[0] // self.patch_size[0])
        
        X = sat_img[:, row_number : row_number + self.patch_size[0], col_number : col_number + self.patch_size[1]] / 255
        
        if self.transform is not None:
            X = self.transform(X)
        
        return X


