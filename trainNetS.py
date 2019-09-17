from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from logger import Logger
import scipy.io as scio
import skimage
from skimage import io
import numpy as np
import argparse

from dataloaderNetS import get_loader
from modelNetS import EncoderNet, ModelNet, EPELoss

parser = argparse.ArgumentParser(description='GeoNetS')
parser.add_argument('--dataset_type', type=int, default=0, metavar='N')
parser.add_argument('--batch_size', type=int, default=32, metavar='N')
parser.add_argument('--epochs', type=int, default= 8, metavar='N')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
parser.add_argument("--dataset_dir", type=str, default='/home/xliea/GeoProj/Dataset/Dataset_256')
args = parser.parse_args()

if(args.dataset_type == 0):
    distortion_type = ['barrel']
elif(args.dataset_type == 1):
    distortion_type = ['pincushion']
elif(args.dataset_type == 2):
    distortion_type = ['rotation']
elif(args.dataset_type == 3):
    distortion_type = ['shear']
elif(args.dataset_type == 4):
    distortion_type = ['projective']
elif(args.dataset_type == 5):
    distortion_type = ['wave']

use_GPU = torch.cuda.is_available()

train_loader = get_loader(distortedImgDir = '%s%s' % (args.dataset_dir, '/train/distorted'),
                  flowDir   = '%s%s' % (args.dataset_dir, '/train/uv'), 
                  batch_size = args.batch_size,
                  distortion_type = distortion_type)

val_loader = get_loader(distortedImgDir = '%s%s' % (args.dataset_dir, '/test/distorted'),
                flowDir   = '%s%s' % (args.dataset_dir, '/test/uv'), 
                batch_size = args.batch_size,
                distortion_type = distortion_type)

model_1 = EncoderNet([1,1,1,1,2])
model_2 = ModelNet(distortion_type[0])
criterion = EPELoss()

print('dataset type:',distortion_type)
print('batch size:', args.batch_size)
print('epochs:', args.epochs)
print('lr:', args.lr)
print('train_loader',len(train_loader))
print('val_loader', len(val_loader))
print(model_1)
print(model_2)
print(criterion)
print(torch.cuda.is_available())

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_1 = nn.DataParallel(model_1)

if torch.cuda.is_available():
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()
    criterion = criterion.cuda()

lr = args.lr
optimizer = torch.optim.Adam(model_1.parameters(), lr=lr)

# Set the logger
step = 0
logger = Logger('./logs')

model_1.train()
model_2.train()
for epoch in range(args.epochs):
    for i, (disimgs, disx, disy) in enumerate(train_loader):
        
        if use_GPU:
            disimgs = disimgs.cuda()
            disx = disx.cuda()
            disy = disy.cuda()
        
        disimgs = Variable(disimgs)
        labels_x = Variable(disx)
        labels_y = Variable(disy)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        
        flow_truth = torch.cat([labels_x, labels_y], dim=1)
        flow_output = model_2(model_1(disimgs))
        loss = criterion(flow_output, flow_truth)

        loss.backward()
        optimizer.step()
        
        print("Epoch [%d], Iter [%d], Loss: %.8f" %(epoch + 1, i + 1, loss.data[0].item()))

        #============ TensorBoard logging ============#
        step = step + 1
        #Log the scalar values
        info = {'loss': loss.data[0]}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

    # Decaying Learning Rate
    if (epoch + 1) % 2 == 0:
        lr /= 2
        optimizer = torch.optim.Adam(model_1.parameters(), lr=lr) 
        
        
torch.save(model_1.state_dict(), '%s%s%s%s' % (distortion_type[0],'_', args.lr, '_model_1.pkl')) 
torch.save(model_2.state_dict(), '%s%s%s%s' % (distortion_type[0],'_', args.lr, '_model_2.pkl')) 

# Test
model_1.eval()
model_2.eval()
total = 0
for i, (disimgs, disx, disy) in enumerate(val_loader):

    if use_GPU:
        disimgs = disimgs.cuda()
        disx = disx.cuda()
        disy = disy.cuda()

    disimgs = Variable(disimgs)
    labels_x = Variable(disx)
    labels_y = Variable(disy)
    
    flow_truth = torch.cat([labels_x, labels_y], dim=1)
    flow_output = model_2(model_1(disimgs))
    loss = criterion(flow_output, flow_truth)
    
    print(loss.data[0].item())
    total = total + loss.data[0]
    
print('test loss',total/(i+1).item())

