# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ###### Importing libraries

import pandas as pd
import rasterio as rio 
import numpy as np
import os
import json
from rasterio.features import rasterize
from osgeo import gdal
from matplotlib import pyplot as plt
from scipy.interpolate import NearestNDInterpolator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import gc
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim import Adam
from torchvision import transforms

import argparse
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score

# #### Data

# FIXME FileNotFoundError: [Errno 2] No such file or directory: 'firescarbiobio128/biobio_final_128_10_03.csv'
# FIXME FileNotFoundError: [Errno 2] No such file or directory: "firescarvalpo128/valparaiso_128_01_02.csv"
biobio_dataset=pd.read_csv("Datasets/bio_val_198.csv")  
valparaiso_dataset=pd.read_csv("Datasets/val_val_196.csv")
dataset=pd.concat([valparaiso_dataset,biobio_dataset], axis=0, ignore_index=True)

LS_max=[3660.5,  4303.5,  4832.0,  6956.0,  6174.5,  6234.0, 1,  1000,  3811.0,  4504.0,  4950.5,
7000.5,  6210.0,  6286.5, 1,  1000]
LI_min= [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  -0.69034,  -605.32933, 0.0,  0.0,  0.0,  0.0,  0.0,
0.0,  -0.7970004,  -573.05183]


def preprocessing(imgdata):
    LS_max=[3660.5,  4303.5,  4832.0,  6956.0,  6174.5,  6234.0, 1,  1000,  3811.0,  4504.0,  4950.5,
    7000.5,  6210.0,  6286.5, 1,  1000]
    LI_min= [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  -0.6903,  -605.3293, 0.0,  0.0,  0.0,  0.0,  0.0,
    0.0,  -0.7970,  -573.0518]
    mean_sprepro=[415.8392, 646.4156, 761.339, 2109.5385, 1843.5207, 1189.6269, 0.4677, 301.9405,
    414.9891, 657.8648, 761.992, 2235.3473, 1853.7414, 1160.7475, 0.4924, 341.9075]
    for k in range(1,17):
        if (imgdata[k-1]>LS_max[k-1]).any():
            if imgdata[k-1].mean()<LS_max[k-1]:
                imgdata[k-1][imgdata[k-1]>LS_max[k-1]]=imgdata[k-1].mean()
            else:
                imgdata[k-1][imgdata[k-1]>LS_max[k-1]]=mean_sprepro[k-1]
        elif (imgdata[k-1]<LI_min[k-1]).any():
            if imgdata[k-1].mean()>LI_min[k-1]:
                imgdata[k-1][imgdata[k-1]<LI_min[k-1]]=imgdata[k-1].mean()
            else: 
                imgdata[k-1][imgdata[k-1]<LI_min[k-1]]=mean_sprepro[k-1]
    return imgdata


# +
class firescardataset():
    """ The objective of this class is to ... """
    def __init__(self,dataset, subset_size1, subset_size2, subset_size3,subset_size4,mult=1,transform=None):
        """ When instantiating this class, the following parameters must be provided:

        Args:
            dataset (type) : purpose
            subset_size1 (type) : purpose
            subset_size2 (type) : purpose
            subset_size3 (type) : purpose
            subset_size4 (type) : purpose
            mult (type) : purpose
            transform (type) : purpose

        :param dataset: dataset with data filenames from two different regions. The first one from the index subset_size1-subset_size2, and the
        second one from subset_size3-subset_size4. There are 3 columns with the required data filenames for each input: 
        "ImPosF"=The image post Fire, "ImgPreF"=The image pre Fire, and "FireScar_tif"=The label, in a raster file"""
        self.transform = transform
        # list of image files (pre and post fire), and labels
        # label vector edge coordinates
        self.imgfiles = []
        self.imgprefiles=[]
        self.labels = []
        self.seglabels = []
        imgposfiles = []
        # read in segmentation label files
        # WARNING demasiados strings hardcoded!!
        for i in range(subset_size1,subset_size2):
            segdata = os.path.join("firescarvalpoallsizes/FireScar/", dataset.loc[i,"FireScar_tif"])
            self.seglabels.append(segdata)
            self.imgfiles.append(os.path.join("firescarvalpo128/ImgPosF/",dataset.loc[i,"ImgPosF"]))
            self.imgprefiles.append(os.path.join("firescarvalpo128/ImgPreF/",dataset.loc[i,"ImgPreF"]))
        for i in range(subset_size3,subset_size4):
            self.seglabels.append(os.path.join("firescarbiobioallsizes/FireScar/",dataset.loc[i,"FireScar_tif"]))
            self.imgfiles.append(os.path.join("firescarbiobio128/ImgPosF/",dataset.loc[i,"ImgPosF"]))
            self.imgprefiles.append(os.path.join("firescarbiobio128/ImgPreF/",dataset.loc[i,"ImgPreF"]))
        self.imgfiles = np.array(self.imgfiles)
        self.imgprefiles=np.array(self.imgprefiles)
        self.labels = np.array(self.labels)
        
        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.imgprefiles = np.array([*self.imgprefiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.seglabels = self.seglabels * mult
        
    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, idx):
        idx=idx-1
        imgfile = rio.open(self.imgfiles[idx])
        imgpre=rio.open(self.imgprefiles[idx])
        imgdata1 = np.array([imgfile.read(i) for i in [1,2,3,4,5,6,7,8]])
        imgdatapre=np.array([imgpre.read(i) for i in [1,2,3,4,5,6,7,8]])
        imgdata=np.concatenate((imgdata1, imgdatapre), axis=0)
        imgdata[imgdata==0]=np.nan

        if (np.isfinite(imgdata)==False).any():                      #Replace nan for the neighbours mean values
            mask=np.where(np.isfinite(imgdata))
            interp=NearestNDInterpolator(np.transpose(mask), imgdata[mask])
            imgdata=interp(*np.indices(imgdata.shape))

        ds = gdal.Open(self.seglabels[idx])
        myarray = np.array(ds.GetRasterBand(1).ReadAsArray())

        x=imgdata1.shape[1]
        y=imgdata1.shape[2]

      #FireScar padding to 128 in case is not that size
        x,y=myarray.shape
                                                                    #only to equalize to 128x128 images or it could be to image size 
        ulx_i, lry_i, lrx_i, uly_i=imgfile.bounds
        ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()
        lrx = ulx + (ds.RasterXSize * xres)
        lry = uly + (ds.RasterYSize * yres)
        left=round((ulx-ulx_i)/xres)    #np.pad(a, up, down, left, right)
        right=round((lrx_i-lrx)/xres)
        up=round((uly-uly_i)/yres)
        down=round((lry_i-lry)/yres)
        myarray=np.pad(myarray,((up, down),(left,right)),"constant")

        imgdata=preprocessing(imgdata)                               #preprocessing to the data when there are values off range (i.e outliers)

        sample = {'idx': idx,
              'img': imgdata,
              'fpt': myarray,
              'imgfile': self.imgfiles[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
#         """
#         :param sample: sample to be converted to Tensor
#         :return: converted Tensor sample
#         """
        out = {'idx': sample['idx'],
        'img': torch.from_numpy(sample['img'].copy()),
        'fpt': torch.from_numpy(sample['fpt'].copy()),
        'imgfile': sample['imgfile']}

        return out
class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
       90 deg, (horizontal) mirroring, and (vertical) flipping."""

    def __call__(self, sample):
        """
        :param sample: sample to be randomized
        :return: randomized sample
        """
        imgdata = sample['img']
        fptdata = sample['fpt']
        idx=sample["idx"]
        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
            fptdata = np.flip(fptdata, 1)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
            fptdata = np.flip(fptdata, 0)
        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        if rot:
            imgdata = np.rot90(imgdata, rot, axes=(1,2))
            fptdata = np.rot90(fptdata, rot, axes=(0,1))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'fpt': fptdata.copy(),
                'imgfile': sample['imgfile']}

class Normalize(object):
    """Normalize pixel values to the range [0, 1] measured using minmax-scaling"""
    def __init__(self):

        self.channel_min=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6737967729568481, -605.4163208007812,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7714285850524902, -572.1392822265625])
        
        self.channel_max=np.array([3560.0, 4147.0, 4780.0, 6832.0, 6143.0, 6261.0, 1.0, 1000.0,
                                   3765.0, 4398.0, 4866.0, 6930.0, 6157.0, 6167.0, 1.0, 1000.0])
            
    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample"""
        sample['img'] = (sample['img']-self.channel_min.reshape(
            sample['img'].shape[0], 1, 1))/(self.channel_max.reshape(
            sample['img'].shape[0], 1, 1)-self.channel_min.reshape(
            sample['img'].shape[0], 1, 1))
        return sample 
def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset; uses same input parameters as PowerPlantDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Normalize(),
            Randomize(),
            ToTensor()
           ])
    else:
        data_transforms = None

    data = firescardataset(*args, **kwargs,
                                         transform=data_transforms)
    
    return data


# -

# ##### Modelo

torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# +
# %%capture
class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
#     """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 128)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.down4 = Down(1024, 2048 // factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.outc = OutConv(128, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


model = UNet(n_channels=16, n_classes=1)
model.to(device)
# -

   # start training process
print('running on...', device)


# #### Training

def dice2d(pred, targs):  
    pred = pred.squeeze()
    targs = targs.squeeze()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


# +
def train_model(model, epochs, opt, loss, batch_size):  
    """
    :param model: model instance
    :param epochs: (int) number of epochs to be trained
    :param opt: optimizer instance
    :param loss: loss function instance
    :param batch_size: (int) batch size
    :param mult: (int) augmentation factor that amplifies the data * mult"""

    data_train = create_dataset(dataset=dataset,subset_size1=0,subset_size2=780,subset_size3=975,subset_size4=1815, mult=3)

    data_val = create_dataset(dataset=dataset,subset_size1=780,subset_size2=975,subset_size3=1815,subset_size4=2024, mult=3)
    train_dl = DataLoader(data_train, batch_size=16, num_workers=0,
                        pin_memory=True) #drop_last=True)
    val_dl = DataLoader(data_val, batch_size=16, num_workers=0,
                        pin_memory=True) # drop_last=True)  
    filename=""   # ending of the model filename
    best_model={}
    best_model["val_loss_total"]=100
    best_dc={}
    best_dc["val_DC"]=0
    # start training
    for epoch in range(epochs):
        model.train()
        #metrics 
        dicec_train_acc=[]
        FN_train=[]
        TP_train=[]
        FP_train=[]
#         train_acc_total=0
        train_loss_total = 0
        train_ious = []
        progress = tqdm(enumerate(train_dl), desc="Train Loss: ",
                        total=len(train_dl))
        for i, batch in progress:
            # try:
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().to(device)
                                                                            
            output = model(x)

            # derive binary segmentation map from prediction
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1
            
            # derive IoU values            
            for j in range(y.shape[0]):                                       
                z = jaccard_score(y[j].flatten().cpu().detach().numpy(),        
                          output_binary[j][0].flatten())
                if (np.sum(output_binary[j][0]) != 0 and
                    np.sum(y[j].cpu().detach().numpy()) != 0):
                    train_ious.append(z)
                    TP_train.append((output_binary.squeeze()*y.cpu().detach().numpy().squeeze()).sum())
                    FN_train.append(((output_binary.squeeze()==0) & (y.cpu().detach().numpy().squeeze()==1)).sum())
                    FP_train.append(((output_binary.squeeze()==1) & (y.cpu().detach().numpy().squeeze()==0)).sum())
                    dicec_train_acc.append(dice2d(output_binary,y.cpu().detach().numpy()))

            # derive scalar binary labels on a per-image basis
            y_bin = np.array(np.sum(y.cpu().detach().numpy(),
                                    axis=(1,2)) != 0).astype(int)
            pred_bin = np.array(np.sum(output_binary,
                                      axis=(1,2,3)) != 0).astype(int)

            # derive image-wise accuracy for this batch
#             train_acc_total += accuracy_score(y_bin, pred_bin)
            # derive loss                                                       
            loss_epoch = loss(output, y.unsqueeze(dim=1))
            train_loss_total += loss_epoch.item()
            progress.set_description("Train Loss: {:.4f}".format(
                train_loss_total/(i+1)))

            # learning
            opt.zero_grad()
            loss_epoch.backward()
            opt.step()
            
        # logging
        writer.add_scalar("training DC", np.average(dicec_train_acc),epoch)
        writer.add_scalar("training CE",  np.mean(FP_train)/(np.mean(TP_train)+np.mean(FP_train)), epoch)
        writer.add_scalar("training OE",  np.mean(FN_train)/(np.mean(TP_train)+np.mean(FN_train)), epoch)                         
        writer.add_scalar("training loss", train_loss_total/(i+1), epoch)
        writer.add_scalar("training iou", np.average(train_ious), epoch)
#         writer.add_scalar("training acc", train_acc_total/(i+1), epoch)
        writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], epoch)
        torch.cuda.empty_cache()

        # evaluation
        model.eval()
        val_loss_total = 0
        val_ious = []
#         val_acc_total = 0
        
        dicec_eval_acc=[]
        FN_eval=[]
        TP_eval=[]
        FP_eval=[]
        
        progress = tqdm(enumerate(val_dl), desc="val Loss: ",
                        total=len(val_dl))
                          
        for j, batch in progress:
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().to(device)
            output = model(x)

          # derive loss
            loss_epoch = loss(output, y.unsqueeze(dim=1))
            val_loss_total += loss_epoch.item()

          # derive binary segmentation map from prediction
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1

          # derive IoU values
            ious = []
            for k in range(y.shape[0]):
                z = jaccard_score(y[k].flatten().cpu().detach().numpy(),
                        output_binary[k][0].flatten())
                if (np.sum(output_binary[k][0]) != 0 and 
                    np.sum(y[k].cpu().detach().numpy()) != 0):
                    val_ious.append(z)
                    TP_eval.append((output_binary.squeeze()*y.cpu().detach().numpy().squeeze()).sum())
                    FN_eval.append(((output_binary.squeeze()==0) & (y.cpu().detach().numpy().squeeze()==1)).sum())
                    FP_eval.append(((output_binary.squeeze()==1) & (y.cpu().detach().numpy().squeeze()==0)).sum())
                    dicec_eval_acc.append(dice2d(output_binary,y.cpu().detach().numpy()))
                   
          # derive scalar binary labels on a per-image basis
            y_bin = np.array(np.sum(y.cpu().detach().numpy(),
                                  axis=(1,2)) != 0).astype(int)
            pred_bin = np.array(np.sum(output_binary,
                                      axis=(1,2,3)) != 0).astype(int)

          # derive image-wise accuracy for this batch
#             val_acc_total += accuracy_score(y_bin, pred_bin)
            
            progress.set_description("val Loss: {:.4f}".format(
             val_loss_total/(j+1)))

        # logging
        writer.add_scalar("val DC", np.average(dicec_eval_acc),epoch)
        writer.add_scalar("val CE",  np.mean(FP_eval)/(np.mean(TP_eval)+np.mean(FP_eval)), epoch)
        writer.add_scalar("val OE",  np.mean(FN_eval)/(np.mean(TP_eval)+np.mean(FN_eval)), epoch)
        writer.add_scalar("val loss", val_loss_total/(j+1), epoch)
        writer.add_scalar("val iou", np.average(val_ious), epoch)
#         writer.add_scalar("val acc", val_acc_total/(j+1), epoch)        
        
        print(("Epoch {:d}: train loss={:.3f}, val loss={:.3f}, "
               "train iou={:.3f}, val iou={:.3f}, "
                "DC training={:.3f}, val DC={:.3f}").format(
                   epoch+1, train_loss_total/(i+1), val_loss_total/(j+1),
                   np.average(train_ious), np.average(val_ious),np.average(dicec_train_acc),
                    np.average(dicec_eval_acc)))

        if (val_loss_total/(j+1))<best_model["val_loss_total"]:
            best_model["val_loss_total"]=(val_loss_total/(j+1))
            best_model["epoch"]=epoch
        if (np.average(dicec_eval_acc))>best_dc["val_DC"]:
            best_dc["val_DC"]=np.average(dicec_eval_acc)
            best_dc["epoch"]=epoch
            
#         if epoch % 1 == 0:                                  #uncomment to save the model files
#             torch.save(model.state_dict(),
#             'U_Net/runs/ep{:0d}_lr{:.0e}_bs{:02d}_{:03d}_{}.model'.format(
#                 args.ep, args.lr, args.bs, epoch, filename))

        writer.flush()
        scheduler.step(val_loss_total/(j+1))
        torch.cuda.empty_cache()
    print("best model: epoch (file): {}, val loss: {}".format(best_model["epoch"], best_model["val_loss_total"]))
    print("best model_dc: epoch (file): {}, val dc: {}".format(best_dc["epoch"], best_dc["val_DC"]))

    return model


# +
# setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-f')

parser.add_argument('-ep', type=int, default=25,    
                    help='Number of epochs')
parser.add_argument('-bs', type=int, nargs='?',             
                    default=16, help='Batch size')
parser.add_argument('-lr', type=float,
                    nargs='?', default=0.0001, help='Learning rate')
# parser.add_argument('-mo', type=float,
#                     nargs='?', default=0.7, help='Momentum')    #for SGD optimizer
args = parser.parse_args()


# setup tensorboard writer
writer = SummaryWriter('U_Net/runs/'+"ep{:0d}_lr{:.0e}_bs{:03d}/".format(
    args.ep, args.lr, args.bs))

# initialize loss function
loss = nn.BCEWithLogitsLoss()

# initialize optimizer
# opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo)
opt = optim.Adam(model.parameters(), lr=args.lr)

# initialize scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',
                                                 factor=0.5, threshold=1e-4,
                                                 min_lr=1e-6)
# -

# # run training
# model.load_state_dict(torch.load(
#  "U_Net/runs/ep25_lr1e-03_bs10_mo0.7.model" , map_location=torch.device('cpu')))
train_model(model, args.ep, opt, loss, args.bs)
writer.close()

# Eval

# +
# # model.load_state_dict(model)
# model.load_state_dict(torch.load(
#     "U_net/runs/128_11_03_fixed_fixed/ep25_lr1e-04_bs16_015.model", map_location=torch.device('cpu')))
# -

# #### Evaluation

# +
# FIXME FileNotFoundError: [Errno 2] No such file or directory: "firescarbiobioallsizes/biobio_testf_01_02.csv"
# FIXME FileNotFoundError: [Errno 2] No such file or directory: "firescarvalpoallsizes/valpo_testf_01_02.csv"
evalb=pd.read_csv("Datasets/bio_test_98.csv")
evalv=pd.read_csv("Datasets/val_test_97.csv")

evald=pd.concat([evalv,evalb],axis=0,ignore_index=True)    

# +
# %%capture
np.random.seed(3)
torch.manual_seed(3)

# load data
data_val = create_dataset(dataset=evald,subset_size1=0,subset_size2=50,subset_size3=50,subset_size4=100, mult=1)

batch_size = 1 # 1 to create diagnostic images, any value otherwise
all_dl = DataLoader(data_val, batch_size=batch_size)#, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

dicec_eval_acc=[]
FN_eval=[]
TP_eval=[]
FP_eval=[]
comission=[]
omission=[]
cont=0
model.eval()

# define loss function
loss_fn = nn.BCEWithLogitsLoss()

# run through test data
all_ious = []
# all_accs = []
test_df=pd.DataFrame(columns=["ImgPosF","iou","DC","CE","OE"])
for i, batch in progress:
    x, y = batch['img'].float().to(device), batch['fpt'].float().to(device)
    idx = batch['idx']

    output = model(x).cpu()

    # obtain binary prediction map
    pred = np.zeros(output.shape)
    pred[output >= 0] = 1

    # derive Iou score
    cropped_iou = []
    for j in range(y.shape[0]):
        z = jaccard_score(y[j].flatten().cpu().detach().numpy(),
                          pred[j][0].flatten())
        if (np.sum(pred[j][0]) != 0 and
            np.sum(y[j].cpu().detach().numpy()) != 0):
            cropped_iou.append(z)       
            
    all_ious = [*all_ious, *cropped_iou]
        
    # derive scalar binary labels on a per-image basis
    y_bin = np.array(np.sum(y.cpu().detach().numpy(),
                            axis=(1,2)) != 0).astype(int)
    prediction = np.array(np.sum(pred,
                               axis=(1,2,3)) != 0).astype(int)
    # derive image-wise accuracy for this batch
#     all_accs.append(accuracy_score(y_bin, prediction))

    # derive binary segmentation map from prediction
    output_binary = np.zeros(output.shape)
    output_binary[output.cpu().detach().numpy() >= 0] = 1

    if batch_size == 1:

        if prediction == 1 and y_bin == 1:
            res = 'true_pos'
        elif prediction == 0 and y_bin == 0:
            res = 'true_neg'
        elif prediction == 0 and y_bin == 1:
            res = 'false_neg'
        elif prediction == 1 and y_bin == 0:
            res = 'false_pos'    
        #scores fix
        TP_eval.append((output_binary.squeeze()*y.cpu().detach().numpy().squeeze()).sum())
        FN_eval.append(((output_binary.squeeze()==0) & (y.cpu().detach().numpy().squeeze()==1)).sum())
        FP_eval.append(((output_binary.squeeze()==1) & (y.cpu().detach().numpy().squeeze()==0)).sum())
        dicec_eval_acc.append(dice2d(output_binary,y.cpu().detach().numpy()))
        test_df.loc[cont,"OE"]=FN_eval[cont]/(TP_eval[cont]+FN_eval[cont])
        test_df.loc[cont,"CE"]=FP_eval[cont]/(TP_eval[cont]+FP_eval[cont])
        test_df.loc[cont,"DC"]=dice2d(output_binary,y.cpu().detach().numpy()) 
        test_df.loc[cont,"ImgPosF"]=(batch['imgfile'][0].split("/")[2])
        OE=FN_eval[cont]/(TP_eval[cont]+FN_eval[cont])
        this_iou = jaccard_score(y[0].flatten().cpu().detach().numpy(),
                                 pred[0][0].flatten())
        test_df.loc[i,"iou"]=this_iou        


         # create plot
        f, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(20,20))
        x=x.cpu()
        y=y.cpu()
        
        # false color plot Image prefire
        ax1.imshow(0.2+1.5*(np.dstack([x[0][12], x[0][11], x[0][10]])-np.min([x[0][12].numpy(),
                            x[0][11].numpy(), x[0][10].numpy()]))/(np.max([x[0][12].numpy(),
                            x[0][11].numpy(), x[0][10].numpy()])-np.min([x[0][12].numpy(),
                            x[0][11].numpy(), x[0][10].numpy()])), origin='upper')
        
        ax1.set_title("ImgPreF",fontsize=12)
        ax1.set_xticks([])
        ax1.set_yticks([])
        #Image Pos-Fire
        ax2.imshow(0.2+1.5*(np.dstack([x[0][4], x[0][3], x[0][2]])-np.min([x[0][4].numpy(), 
                            x[0][3].numpy(), x[0][2].numpy()]))/(np.max([x[0][4].numpy(),
                            x[0][3].numpy(), x[0][2].numpy()])-np.min([x[0][4].numpy(),
                            x[0][3].numpy(), x[0][2].numpy()])), origin='upper')
        
        ax2.set_title("ImgPosF",fontsize=12)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # segmentation ground-truth and prediction
        ax3.imshow(y[0], cmap='Greys_r', alpha=1)
        ax4.imshow(pred[0][0], cmap='Greys_r', alpha=1)
        ax3.set_title("Original Scar",fontsize=12)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.annotate("IoU={:.2f}".format(this_iou), xy=(5,15), fontsize=15)

        ax4.set_title({'true_pos': 'Scar Prediction: True Positive \n  -IoU={:.2f},' 
                       '-OE={:.2f}, -CE={:.2f}, -DC={:.2F}'.format(this_iou, test_df.loc[cont,"OE"],test_df.loc[cont,"CE"],test_df.loc[cont,"DC"]),
               'true_neg': 'Scar Prediction: True Negative \n  -IoU={:.2f},' 
            '-OE={:.2f}, -CE={:.2f}, -DC={:.2F}'.format(this_iou,test_df.loc[cont,"OE"],test_df.loc[cont,"CE"],test_df.loc[cont,"DC"]),
               'false_pos': 'Scar Prediction: False Positive   -IoU={:.2f},'
             '-OE={:.2f}, -CE={:.2f}, -DC={:.2F}'.format(this_iou, test_df.loc[cont,"OE"],test_df.loc[cont,"CE"],test_df.loc[cont,"DC"]),
                'false_neg': 'Scar Prediction: False Negative \n  -IoU={:.2f},'
            '-OE={:.2f}, -CE={:.2f}, -DC={:.2F}'.format(this_iou,test_df.loc[cont,"OE"], 0,test_df.loc[cont,"DC"])}[res],
                      fontsize=12)
        
        cont+=1      

        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)

        # plt.savefig("U_Net/output/"+(os.path.split(batch['imgfile'][0])[1]).\
        #             replace('.tif', '.png').replace(':', '_'),
        #              dpi=200)   
        
        plt.close()     #comment to display

print('iou:', len(all_ious), np.average(all_ious))
# -

# ###### Test analysis 

# +
# test_df.to_csv("U_Net/runs/test_df_128_final1103.csv")
# -

test_df["DC"].mean(), test_df["OE"].mean(),test_df["CE"].mean()
