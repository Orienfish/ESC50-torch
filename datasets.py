import librosa
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def spec_to_image(spec,
                  eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    # First standard normalize
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    # Normalize to 0.0-1.0
    spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min + eps)

    # Normalize to 0-255
    # spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    # spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

fig_cnt = {}
def get_melspectrogram_db(file_path,
                          label,
                          sr=None,
                          n_fft=2048,
                          hop_length=512,
                          n_mels=32,
                          fmin=20,
                          fmax=8300,
                          top_db=80,
                          win_secs=5.0, # the duration of one smaple in seconds
                          overlap=0.75, # the overlapping of samples
                          plot=False):   
    # For recording plot numbers
    if label not in fig_cnt:
        fig_cnt[label] = 0

    wav, sr = librosa.load(file_path, sr=sr)
    sample_num = int(sr * win_secs)
    spec_db = []

    # Pad the signal to 5 seconds, as in the original code
    if wav.shape[0] < 5*sr:
        wav = np.pad(wav, int(np.ceil((5*sr-wav.shape[0])/2)), mode='reflect')
    else:
        wav = wav[:5*sr]
    
    # Obtain the sample of every "window" with the defined length
    ind = 0
    while ind + sample_num <= wav.shape[0]:
        cur_wav = wav[ind:ind+sample_num]
        spec = librosa.feature.melspectrogram(y=cur_wav, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels,
                                              fmin=fmin, fmax=fmax)
        cur_spec_db = spec_to_image(librosa.power_to_db(spec, top_db=top_db))[np.newaxis,...]
        spec_db.append(cur_spec_db)

        if plot:
            plt.figure()
            librosa.display.waveshow(cur_wav, sr=sr)
            plt.title(label)
            plt.savefig('temp_plots/{}{}.png'.format(label, fig_cnt[label]))
            plt.close()
            fig_cnt[label] += 1

        ind += int(sample_num * (1-overlap))
    # print(len(spec_db))
    return spec_db


# ESC-50 and ESC-10
# Adapted from https://github.com/hasithsura/Environmental-Sound-Classification/blob/master/ESC50-Pytorch.ipynb
class ESCDataset(Dataset):
    def __init__(self, root: str,
                 esc50: bool=True,
                 val_fold: int=4,
                 train: bool=True,
                 download: bool=True,
                 win_secs: float=5.0,
                 overlap: float=0.75,
                 n_mels: int=64):
        """
        Dataset setup of ESC-50 and ESC-10
        :param root: The root directory of the downloaded dataaset
        :param esc50: True for ESC-50, False for ESC-10
        :param val_fold: fold id for validation
        :param train: True for training, False for validation
        :param download: True for downloading the dataset
        :param win_secs: The duration per sample in seconds, critical for sample size
            and for accuracy
        :param overlap: The overlap of subsequent samples
        :param n_mels: The number of frequency bin s in Mel Spectrogram
        """
        self.data = []
        self.labels = []

        # Download the dataset
        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                if not os.path.exists(root):
                    os.mkdir(root)
                cmd = 'wget -O {} {}'.format(
                    os.path.join(root, 'master.zip'),
                    'https://github.com/karoldvl/ESC-50/archive/master.zip'
                )
                os.system(cmd)
                os.system('unzip -o {} -d {}'.format(
                    os.path.join(root, 'master.zip'),
                    root
                ))

        # Read meta data
        self.df = pd.read_csv(os.path.join(root, 
                                           'ESC-50-master/meta/esc50.csv'))
        if train:  # Training data
            self.df = self.df[self.df['fold'] != val_fold]
        else:  # Validation data
            self.df = self.df[self.df['fold'] == val_fold]
        out_col = 'category'
        in_col = 'filename'
        self.c2i = {}
        self.i2c = {}

        # ESC-50 or ESC-10
        if esc50:
            self.num_classes = 50
        else:  # ESC-10
            self.num_classes = 10
            self.df = self.df[self.df['esc10']]  # Only keep the ESC-10 rows
        self.categories = sorted(self.df[out_col].unique())

        for i, category in enumerate(self.categories):
            self.c2i[category] = i
            self.i2c[i] = category

        folder_path = os.path.join(root, 'ESC-50-master/audio')
        for ind in tqdm(range(len(self.df))):
            row = self.df.iloc[ind]
            file_path = os.path.join(folder_path, row[in_col])
            new_data = get_melspectrogram_db(file_path,
                                             self.c2i[row['category']],
                                             n_mels=n_mels,
                                             win_secs=win_secs,
                                             overlap=overlap)
            new_label = [self.c2i[row['category']]] * len(new_data)
            self.data.extend(new_data)
            self.labels.extend(new_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# BDLib
class BDLibDataset(Dataset):
    def __init__(self, root: str, 
                 fold_ids: list,
                 download: bool=True,
                 win_secs: float=5.0,
                 overlap: float=0.75,
                 n_mels: int=64):
        """
        Dataset setup of BDLib
        :param root: The root directory of the downloaded dataaset
        :param fold_ids: list of integers, fold to use
        :param download: True for downloading the dataset
        :param win_secs: The duration per sample in seconds, critical for sample size
            and for accuracy
        :param overlap: The overlap of subsequent samples
        :param n_mels: The number of frequency bin s in Mel Spectrogram
        """
        self.data = []
        self.labels = []
        self.all_labels = ['airplane', 'alarms', 'applause',
                           'birds', 'dogs', 'motorcycles',
                           'rain', 'rivers', 'seawaves', 'thunders']
        self.num_classes = 10

        # Download the dataset
        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                if not os.path.exists(root):
                    os.mkdir(root)
                cmd = 'wget -O {} {}'.format(
                    os.path.join(root, 'BDLib.zip'),
                    'http://research.playcompass.com/files/BDLib-2.zip'
                )
                os.system(cmd)
                os.system('unzip -o {} -d {}'.format(
                    os.path.join(root, 'BDLib.zip'),
                    root
                ))

        # Read all samples
        for id in fold_ids:
            dir_path = os.path.join(root, 'fold-{}'.format(id))
            all_files = os.listdir(dir_path)
            for ind in range(len(all_files)):
                file_path = os.path.join(dir_path, all_files[ind])
                label = all_files[ind].split('.')[0].rstrip('0123456789')
                new_data = get_melspectrogram_db(file_path,
                                                 label,
                                                 n_mels=n_mels,
                                                 win_secs=win_secs,
                                                 overlap=overlap)
                new_label = [self.all_labels.index(label)] * len(new_data)
                #print(len(new_data))
                #print(new_label)
                self.data.extend(new_data)
                self.labels.extend(new_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    if sys.argv[1] == 'esc10':
        train_data = ESCDataset(root='ESC50', 
                                esc50=False, 
                                val_fold=4,
                                train=True,
                                download=True,
                                win_secs=5.0,
                                plot=True)
    elif sys.argv[1] == 'esc50':
        train_data = ESCDataset(root='ESC50', 
                                esc50=True, 
                                val_fold=4,
                                train=True,
                                download=True,
                                win_secs=5.0,
                                plot=True)
    elif sys.argv[1] == 'bdlib':
        train_data = BDLibDataset(root='BDLib', 
                                  fold_ids=[1,2,3],
                                  download=True,
                                  plot=True)
    print(train_data.data[0].shape)
    print(train_data.labels[0])
        