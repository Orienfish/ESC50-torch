import librosa
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from tqdm import tqdm


def spec_to_image(spec,
                  eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def get_melspectrogram_db(file_path,
                          sr=None,
                          n_fft=2048,
                          hop_length=512,
                          n_mels=128,
                          fmin=20,
                          fmax=8300,
                          top_db=80):
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0] < 5*sr:
        wav = np.pad(wav, int(np.ceil((5*sr-wav.shape[0])/2)), mode='reflect')
    else:
        wav = wav[:5*sr]
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels,
                                          fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


# ESC-50 and ESC-10
# Adapted from https://github.com/hasithsura/Environmental-Sound-Classification/blob/master/ESC50-Pytorch.ipynb
class ESCDataset(Dataset):
    def __init__(self, root: str,
                 esc50: bool=True,
                 val_fold: int=4,
                 train: bool=True,
                 download: bool=True):
        """
        Dataset setup of ESC-50 and ESC-10
        :param root: The root directory of the downloaded dataaset
        :param esc50: True for ESC-50, False for ESC-10
        :param val_fold: fold id for validation
        :param train: True for training, False for validation
        :param download: True for downloading the dataset
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
                    os.path.join(root, 'HAR.zip'),
                    'https://github.com/karoldvl/ESC-50/archive/master.zip'
                )
                os.system(cmd)
                os.system('unzip -o {} -d {}'.format(
                    os.path.join(root, 'master.zip'),
                    root
                ))

        # Read meta data
        self.df = pd.read_csv(os.path.join(root, 'meta/esc50.csv'))
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

        for ind in tqdm(range(len(self.df))):
            row = self.df.iloc[ind]
            file_path = os.path.join(root + '/audio', row[in_col])
            self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis, ...])
            self.labels.append(self.c2i[row['category']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# BDLib
class BDLibData(Dataset):
    def __init__(self, root, fold_ids):
        self.data = []
        self.labels = []
        self.all_labels = ['airplane', 'alarms', 'applause',
                           'birds', 'dogs', 'motorcycles',
                           'rain', 'rivers', 'seawaves', 'thunders']
        self.num_classes = 10

        # Read all samples
        for id in fold_ids:
            dir_path = os.path.join(root, 'fold-{}'.format(id))
            all_files = os.listdir(dir_path)
            for ind in range(len(all_files)):
                file_path = os.path.join(dir_path, all_files[ind])
                self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis,...])
                label = all_files[ind].split('.')[0].rstrip('0123456789')
                # print(label)
                self.labels.append(self.all_labels.index(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
