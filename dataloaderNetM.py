import os
from torch.utils import data
from torchvision import transforms
import scipy.io as spio
import numpy as np
import skimage
import torch

"""Custom Dataset compatible with prebuilt DataLoader."""
class DistortionDataset(data.Dataset):
    def __init__(self, distortedImgDir, flowDir, transform, distortion_type, data_num):
        
        self.distorted_image_paths = []
        self.displacement_paths = []
            
        for fs in os.listdir(distortedImgDir):
            
            types = fs.split('_')[0]
            number = int(fs.split('_')[1].split('.')[0])
            
            if types in distortion_type and number < data_num:
                self.distorted_image_paths.append(os.path.join(distortedImgDir, fs)) 
        
        for fs in os.listdir(flowDir):
            
            types = fs.split('_')[0]
            number = int(fs.split('_')[1].split('.')[0])
            
            if types in distortion_type and number < data_num:
                self.displacement_paths.append(os.path.join(flowDir, fs)) 

        self.distorted_image_paths.sort()
        self.displacement_paths.sort()
        
        self.transform = transform
        
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.distorted_image_paths[index]
        displacement_path = self.displacement_paths[index]
        
        distorted_image =skimage.io.imread(distorted_image_path)
        displacement = spio.loadmat(displacement_path)

        displacement_x = displacement['u'].astype(np.float32)
        displacement_y = displacement['v'].astype(np.float32)
        
        displacement_x = displacement_x[np.newaxis,:]
        displacement_y = displacement_y[np.newaxis,:]

        label_type = distorted_image_path.split('_')[0].split('/')[-1]
        label = 0
        if (label_type == 'barrel'):
            label = 0
        elif (label_type == 'pincushion'):
            label = 1
        elif (label_type == 'rotation'):
            label = 2
        elif (label_type == 'shear'):
            label = 3
        elif (label_type == 'projective'):
            label = 4
        elif (label_type == 'randomnew'):
            label = 5

        if self.transform is not None:
            trans_distorted_image = self.transform(distorted_image)
        else:
            trans_distorted_image = distorted_image
   
        return trans_distorted_image, displacement_x, displacement_y, label

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.distorted_image_paths)
    
    
def get_loader(distortedImgDir, flowDir, batch_size, distortion_type, data_num):
    """Builds and returns Dataloader."""
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = DistortionDataset(distortedImgDir, flowDir, transform, distortion_type, data_num)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return data_loader