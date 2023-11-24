# %%
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import random
import torchhd
from torchhd.models import Centroid
import torchmetrics
import tensorboard_logger as tb_logger

# Import self-implemented torch dataset
from datasets import BDLibDataset, ESCDataset
from torchaudio.datasets import SPEECHCOMMANDS

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='batch_size in validation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs or number of passes on dataset')
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')
    
    # HDC encoder
    parser.add_argument('--hd_encoder', type=str, default='timeseries',
                        choices=['none', 'rp', 'idlevel', 'nonlinear', 'timeseries'],
                        help='the type of hd encoding function to use')
    parser.add_argument('--dim', type=int, default=1000,
                        help='the size of HD space dimension')
    parser.add_argument('--levels', type=int, default=100,
                        help='the number of quantized level used on raw data')
    parser.add_argument('--flipping', type=float, default=0.01,
                        help='the flipping rate in the time series encoder')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='esc10',
                        choices=['esc10', 'esc50', 'bdlib', 'speechcommands'],
                        help='dataset')
    parser.add_argument('--win_secs', type=float, default=5.0,
                        help='window size for wav data')
    parser.add_argument('--overlap', type=float, default=0.75,
                        help='the ratio of overlap in generating sliding window')
    parser.add_argument('--n_mels', type=int, default=64,
                        help='the number of frequency bins in mel spectrogram')
    

    opt = parser.parse_args()

    # set the path according to the environment
    opt.tb_path = './tb_results/{}_tensorboard/'.format(opt.dataset)
    if not os.path.isdir(opt.tb_path):
        os.makedirs(opt.tb_path)

    opt.model_name = '{}_{}_{}_{}_hd{}_{}_{}_{}_bsz{}_{}_epoch{}_trial{}'.format(
        opt.dataset, opt.win_secs, opt.overlap, opt.n_mels,
        opt.dim, opt.hd_encoder, opt.levels, opt.flipping, 
        opt.batch_size, opt.val_batch_size, opt.epochs, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


# HDC timeseries encoder
class timeseries_Encoder():
    def __init__(self,
                 feat_num,
                 quantization_num,
                 D,
                 P,
                 min=0.0,
                 max=1.0):
        self.feat_num = feat_num
        self.quantization_num = int(quantization_num)
        self.D = int(D)
        self.P = float(P)
        self.min = float(min)
        self.max = float(max)
        self.range = max - min
        self.init_hvs()

    def init_hvs(self):
        # level hvs
        num_flip = int(self.D * self.P)
        self.level_hvs = [np.random.randint(2, size=self.D)]
        for i in range(self.quantization_num-1):
            new = copy.deepcopy(self.level_hvs[-1])
            idx = np.random.choice(self.D,num_flip,replace=False)
            new[idx] = 1-new[idx]
            self.level_hvs.append(new)
        self.level_hvs = np.stack(self.level_hvs)

        #id hvs
        self.id_hvs = []
        for i in range(self.feat_num):
            self.id_hvs.append(np.random.randint(2, size=self.D))
        self.id_hvs = np.stack(self.id_hvs)

    def quantize(self, one_sample):
        quantization = self.level_hvs[((((one_sample - self.min) / self.range) * self.quantization_num) - 1).astype('i')]
        return quantization

    def bind(self,a,b):
        return np.logical_xor(a,b).astype('i')

    def permute(self,a):
        for i in range(len(a)):
            a[i] = np.roll(a[i],i,axis=1)
        return a

    def sequential_bind(self,a):
        return np.sum(a,axis=0) % 2

    def bipolarize(self,a):
        a[a==0] = -1
        return a

    def encode_one_time_series_sample(self, one_sample):
        one_sample = one_sample.cpu().numpy()
        T = len(one_sample)
        out = self.quantize(one_sample)
        out = self.bind(out,np.repeat(np.expand_dims(self.id_hvs,0),T,0))
        out = self.permute(out)
        out = self.sequential_bind(out)
        out = self.bipolarize(out)
        out = np.sum(out,axis=0)
        return torch.from_numpy(out).float()


# HDC encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, levels, hd_dim, flipping, device):
        """
        :param input_dim: dimension of the number of sensors
        :param levels: number of quantized levels
        :param hd_dim: HDC dimension
        :param flipping: flipping ratio to generate the next level hypervector
        :param device: cuda or cpu within torch
        """
        super(Encoder, self).__init__()
        self.device = device
        self.wav_enc = timeseries_Encoder()
        self.spec_enc = timeseries_Encoder(input_dim, levels, hd_dim, flipping)

    def forward(self, x):
        x = x.squeeze()
        enc = []
        batch_size = x.shape[0]
        for i in range(batch_size):
            enc.append(
                self.timeseries_Encoder.encode_one_time_series_sample(x[i]).to(self.device))
        sample_hv = torch.stack(enc, dim=0)
        return torchhd.hard_quantize(sample_hv)
    

# HDC training
def hd_train(train_loader, model, encode, epoch, device):
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)
    
            # After the following steps, the input data shape should be (batch_size, num_of_time_steps, num_of_sensors)
            samples = samples.squeeze()  # First get rid of the dimension with length=1
            samples = samples.swapaxes(1,2)  # Temp fix for BDLib, need to switch the time and frequency axis
            # print('input sample shape', samples.shape)
            
            samples_hv = encode(samples)
            model.add(samples_hv, labels)
            if epoch > 0: # Retraining
                predict = torch.argmax(model(samples_hv, dot=True), dim=-1)
                wrong_pred = (predict != labels)
                model.add(-samples_hv[wrong_pred], predict[wrong_pred])
                model.add(samples_hv[wrong_pred], labels[wrong_pred])

                    
# HDC testing
def hd_test(valid_loader, model, encode, device):
    true_labels, pred_labels = [], []
    with torch.no_grad():
        model.normalize()
    
        for samples, labels in tqdm(valid_loader, desc="Testing"):
            samples = samples.to(device)
    
            # After the following steps, the input data shape should be (batch_size, num_of_time_steps, num_of_sensors)
            samples = samples.squeeze()  # First get rid of the dimension with length=1
            samples = samples.swapaxes(1,2)  # Temp fix for BDLib, need to switch the time and frequency axis
            
            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            
            true_labels.extend(labels.tolist())
            pred_labels.extend(torch.argmax(outputs, dim=1).tolist())
            
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    acc = (true_labels == pred_labels).sum() / true_labels.size
    acc_pc = []
    for c in np.unique(true_labels):
        mask = (true_labels == c)
        acc_pc.append((true_labels[mask] == pred_labels[mask]).sum() / mask.sum())
    print(f"Testing accuracy of {acc}%")
    print("Testing accuracy per class: ", acc_pc)
    return acc, acc_pc


def main():
    opt = parse_option()
    
    print("============================================")
    print(opt)
    print("============================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    # set seed for reproducing
    random.seed(opt.trial)
    np.random.seed(opt.trial)
    torch.manual_seed(opt.trial)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # setup dataset
    if opt.dataset == 'esc10':
        train_data = ESCDataset(root='ESC50', 
                                esc50=False, 
                                val_fold=4,
                                train=True,
                                download=True,
                                win_secs=opt.win_secs,
                                overlap=opt.overlap,
                                n_mels=opt.n_mels)
        valid_data = ESCDataset(root='ESC50', 
                                esc50=False, 
                                val_fold=4,
                                train=False,
                                download=False,
                                win_secs=opt.win_secs,
                                overlap=opt.overlap,
                                n_mels=opt.n_mels)
    elif opt.dataset == 'esc50':
        train_data = ESCDataset(root='ESC50', 
                                esc50=True, 
                                val_fold=4,
                                train=True,
                                download=True,
                                win_secs=opt.win_secs,
                                overlap=opt.overlap,
                                n_mels=opt.n_mels)
        valid_data = ESCDataset(root='ESC50', 
                                esc50=True, 
                                val_fold=4,
                                train=False,
                                download=False,
                                win_secs=opt.win_secs,
                                overlap=opt.overlap,
                                n_mels=opt.n_mels)
    elif opt.dataset == 'bdlib':
        train_data = BDLibDataset(root='BDLib', 
                                 fold_ids=[1,2],
                                 download=True,
                                 win_secs=opt.win_secs,
                                 overlap=opt.overlap,
                                 n_mels=opt.n_mels)
        valid_data = BDLibDataset(root='BDLib', 
                                 fold_ids=[3],
                                 download=False,
                                 win_secs=opt.win_secs,
                                 overlap=opt.overlap,
                                 n_mels=opt.n_mels)
    elif opt.dataset == 'speechcommands':
        train_data = SPEECHCOMMANDS(root='SpeechCommands',
                                    download=True,
                                    subset='training')
        valid_data = SPEECHCOMMANDS(root='SpeechCommands',
                                    download=False,
                                    subset='testing')

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=opt.val_batch_size, shuffle=True)

    print('train data length', len(train_data))
    print('valid data length', len(valid_data))
    if opt.dataset != 'speechcommands':  # Non speechcommands
        print('Sample shape', train_data.data[0].shape)
    else:  # Speechcommands
        print(train_data.__getitem__(0)[0].shape)  # (1,16000)

    # accuracy metric
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=train_data.num_classes)

    encode = Encoder(opt.n_mels, opt.levels, opt.dim, opt.flipping, device)
    encode = encode.to(device)

    model = Centroid(opt.dim, train_data.num_classes)
    model = model.to(device)

    for epoch in range(opt.epochs):
        hd_train(train_loader, model, encode, epoch, device)
        acc, _ = hd_test(valid_loader, copy.deepcopy(model), encode, accuracy, device)
        logger.log_value('accuracy', acc, epoch)


if __name__ == '__main__':
    main()



