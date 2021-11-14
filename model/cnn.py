import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Sequence
#from typeguard import check_argument_types
class Convblock(nn.Module):
    '''
    conv block
    '''
    def __init__(self,conv_in_chans,conv_out_chans,conv_kernel_size,conv_stride,pool,padding) -> None:
        super(Convblock, self).__init__()
        self.covb = torch.nn.Sequential(
                torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False,
                    ),
                torch.nn.MaxPool2d(pool),
                torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout2d(p=0.3))
    
    def forward(self,x):
        return self.covb(x)
    
class CNN_model(nn.Module):
    '''
    Only cnn based model
    '''
    def __init__(self):
        super(CNN_model,self).__init__()
        conv1 = Convblock(1,8,(5,5),(2,2),(2,1),(1,1))
        conv2 = Convblock(8,16,(5,5),(2,2),(2,1),(1,1))
        conv3 = Convblock(16,32,(5,5),(2,2),(2,1),(1,1))
        #making conv block 
        self.conv = torch.nn.Sequential(conv1,conv2,conv3)
        fc1 = torch.nn.Linear(512,128)
        fc2 = torch.nn.Linear(128,32)
        fc3 = torch.nn.Linear(32,8)
        #making linear block 
        self.linear = torch.nn.Sequential(fc1,torch.nn.ReLU(),torch.nn.Dropout(p=0.4),fc2,torch.nn.ReLU(),torch.nn.Dropout(p=0.4),fc3)
        
    def forward(self,x,length):
        conv_out = self.conv(x)
        out = conv_out.reshape(conv_out.size(0),-1)
        out = self.linear(out)
        return out
    