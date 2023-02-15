import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from evaluate import evaluate

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from unet import UNet
from unet import MultiResUnet
import os
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print("Format: ntrain.py modelname or/aug device")
        sys.exit(0)
    
    modelname = sys.argv[1]
    mode = sys.argv[2]
    devicenum = sys.argv[3]
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = devicenum
    
    
    if modelname == "unet":
        model = UNet(n_channels=1, n_classes=2)
        # dir_checkpoint = Path('./checkpoints/')
    elif modelname == "multires":
        model = MultiResUnet(input_channels=1, num_classes=2)
    else:
        print("Wrong model info! Acceptable - unet/multires")
        sys.exit(0)
    
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n')
    model = model.to(memory_format=torch.channels_last)
    model = model.to(device=device)
    
    if mode == "or":
        dir_img = Path('./data/imgs_original/')
        dir_mask = Path('./data/masks_original/')
        dir_checkpoint = Path('./checkpoint_original/')
    elif mode == "aug":
        dir_img = Path('./data/imgs/')
        dir_mask = Path('./data/masks/')
        dir_checkpoint = Path('./checkpoint_augment/')
    
    epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 1e-3
    val_percent: float = 0.1
    save_checkpoint: bool = True
    img_scale: float = 0.25
    amp: bool = False
    weight_decay: float = 1e-8
    momentum: float = 0.999
    gradient_clipping: float = 1.0
    
    
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    opt = optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    step=0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        logging.info("Epoch - {}/{}".format(epoch+1,epochs))
        model.train()
        train_loss_avg=0

        for batch in tqdm(train_loader):
            frame, mask = batch['image'], batch['mask']
            frame=frame.to(device=device,dtype=torch.float32)
            mask=mask.to(device=device,dtype=torch.long)
            pred=model(frame)
            loss=criterion(pred, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_avg += loss.item()
            step+=1
            # update()
        train_loss_avg /= len(train_loader)
        train_losses.append(train_loss_avg/len(train_loader))
        
        logging.info("Train Loss - {}".format(train_loss_avg)) 
        
        model.eval()
        val_loss_avg=0
        n_val=0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                frame, mask = batch['image'], batch['mask']
                frame=frame.to(device=device,dtype=torch.float32)
                mask=mask.to(device=device,dtype=torch.long)
                pred=model(frame)
                loss=criterion(pred,mask)
                val_loss_avg+=loss.item()
                n_val+=1
        val_loss_avg=val_loss_avg/n_val
        # t_vals.append(step)
        val_losses.append(val_loss_avg)
        logging.info("Validation Loss - {}".format(val_loss_avg)) 
        
       

        torch.save(model.state_dict(),os.path.join(dir_checkpoint,"{}_{epoch:d}_{val_loss:.2e}.pth"
                                                   .format(modelname,epoch=epoch+1,val_loss=val_loss_avg)))
    
    with open("loss_direc/{}_{}_trainingLoss".format(modelname,mode),"wb") as fp:
            pickle.dump(train_losses, fp)

    with open("loss_direc/{}_{}_validLoss".format(modelname,mode),"wb") as fp:
        pickle.dump(val_losses, fp)
        
    
    