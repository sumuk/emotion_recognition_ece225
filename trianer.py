import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
from utils.dataset import Audiodataset
import argparse
from pathlib import Path
import logging 
import datetime
import time
import json
from model.cnn import CNN_model 
parser = argparse.ArgumentParser()

parser.add_argument('train_dataset',type=Path,help='path for the train data')
parser.add_argument('test_dataset',type=Path,help='path for test data')
parser.add_argument('--experiment_name',type=str,default='exp')
parser.add_argument('--learning_rate',type=float,default=0.005,help='learning rate for training')
parser.add_argument('--sample_per_emotion',type=int,default=10,help='number of sample per emotion class')
parser.add_argument('--save_model',type=int,default=50,help='number of iteration to save model')
parser.add_argument('--folder_path',type=Path,default='saved_model',help='folder to save the model')
parser.add_argument('--log_file',type=bool,default=True,help='flag to save the log in exp folder')
parser.add_argument('--no_of_iteration',type=int,default=100,help='number of iteration before terminating')
parser.add_argument('--model_selection',type=str,default='cnn',help='model for training')
parser.add_argument('--test_iteration',type=int,default=10,help='number of iteration to test the model')

args = parser.parse_args()

assert os.path.isdir(args.train_dataset),'path for train data doesnt exist'
assert os.path.isdir(args.test_dataset),'path for test data doesnt exist'



if not os.path.isdir(os.path.join(args.experiment_name,args.folder_path)):
    os.makedirs(os.path.join(args.experiment_name,args.folder_path))
if args.log_file:
    logging.basicConfig(
                        filename='{1}/log_{0}.txt'.
                        format(datetime.date.fromtimestamp(time.time()).strftime('%d_%m_%y'),args.experiment_name),
                        filemode='w',
                        level=logging.INFO)

logging.info('loading train dataset')
train_dataset = Audiodataset(args.train_dataset,args.sample_per_emotion)
logging.info('loading test dataset')
test_dataset = Audiodataset(args.test_dataset,args.sample_per_emotion)


# writing the label config in the exp folder 
with open('{0}/mapping.json'.format(args.experiment_name),'w') as ftp:
    json.dump(train_dataset.get_mapping(),ftp)



train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
 )

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
 )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
best_loss = 1000
loss_function = nn.CrossEntropyLoss()

for iteration_no in range(args.no_of_iteration):
    loss_loop=[]
    for data,label,length in train_loader:
        data = data.transpose(0,1).contiguous()
        optimizer.zero_grad()
        out = model(data.to(device),length.to(torch.long).to(device))
        loss = loss_function(out,label.reshape(-1).to(device))
        loss.backward()
        optimizer.step()
        loss_loop.append(loss.item())
    if iteration_no%args.save_model==0:
        torch.save({
            'epoch': iteration_no,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(loss_loop),
            }, '{0}/{1}/model_{2}.pt'.format(args.experiment_name,args.folder_path,iteration_no))
    if iteration_no%args.test_iteration==0:
        model.eval()
        loss_test=[]
        for data,label,length in test_loader: 
            data = data.transpose(0,1).contiguous()
            out = model(data.to(device),length.to(torch.long).to(device))
            loss = loss_function(out,label.reshape(-1).to(device))
            loss_test.append(loss.item())
            
        logging.info('iteration {1} loss train {0} loss test {2}'.format(np.mean(loss_loop),iteration_no,np.mean(loss_test)))
        model.train()
        if np.mean(loss_test)<best_loss:
            torch.save({
            'epoch': iteration_no,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(loss_test),
            }, '{0}/{1}/model_best.pt'.format(args.experiment_name,args.folder_path))
            best_loss=np.mean(loss_test)
            

    



    

