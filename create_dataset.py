import os
import glob
import time
import multiprocessing

import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav

import librosa
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset



PATH = "audio_S18k/"        # Modify here for your audio directory path.

DATA_PATH = "dataloaders/"
MODEL_PATH = "models/"
GRAPHS_PATH = "graphs/"
CSV_PATH = "csv/"

list_folders = [DATA_PATH, GRAPHS_PATH, MODEL_PATH, CSV_PATH]

for folder in list_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"{folder} folder created.")

MULTIPROCESS = False # False on Windows, multiprocessing does not seem to be supported.

if MULTIPROCESS:
    NUM_CORE = multiprocessing.cpu_count()
    print(f'core numbers {NUM_CORE}')


OLD_TRAIN_LABELS_PATH = 'evaluation_setup/fold1_train.csv'

TRAIN_LABELS_PATH = 'evaluation_setup/train.csv'
VAL_LABELS_PATH = 'evaluation_setup/val.csv'
TEST_LABELS_PATH = 'evaluation_setup/fold1_test.csv'

ORIGINAL_DURATION = 10
AUDIO_DURATION = 4
SAMPLE_RATE = 18000

DICT_ENCODE_CLASS = {0: 'outdoor', 1: 'indoor', 2: 'transportation'}
DICT_CLASS_ENCODE = {'outdoor': 0, 'indoor': 1, 'transportation': 2}

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class DCASEDataset(Dataset):
    def __init__(self, list_names, target, path, transform='', mode='train'):
        # super(DCASEDataset, self).__init__()
        self._dataset_kind = None
        self.file_names = list_names
        self.mode = mode
        self.transform = True if transform == 'random_crop' else False
        self.path = path

        self.sr = SAMPLE_RATE
        self.audio_duration = AUDIO_DURATION

        self.filters = self.generate_filters()

        if self.mode == 'test':
            self.classe = {'airport': 1, 'shopping_mall': 1, 
                            'metro_station': 1, 'street_pedestrian': 0,
                            'public_square': 0, 'street_traffic': 0 ,
                            'tram': 2, 'bus': 2, 'metro': 2, 'park': 0}

        elif self.mode == 'train':
            self.classe = DICT_CLASS_ENCODE
            self.label = target

    def generate_filters(self):
        """
        Generate filters for Data Augmentation
        """
        filters = []

        # HP&LP
        transition = 300
        cutoff = [340, 1000, 2000, 4000, 1500]
        size = 200
        for i, k in enumerate(cutoff):
            params = [0, k, k + transition, self.sr/2]
            filters.append(sig.remez(size, params, [1, 0], Hz=self.sr))
            params = [0, k - 250, k , self.sr/2]
            filters.append(sig.remez(125, params, [0, 1], Hz=self.sr))

        # BP
        transition = 300
        cutoff = [[340, 3400], [1200, 3400]]
        size = 200
        for i, band in enumerate(cutoff):
            params = [0, band[0] - transition, band[0], band[1], band[1] + transition, self.sr/2]
            filters.append(sig.remez(size, params, [0, 1, 0], Hz=self.sr))

        return filters

    def random_crop(self, x):
        """
        Randomly take {self.audio_duration} seconds of the audio
        """
        precision = int(self.sr/5)
        audio_duration = torch.abs(torch.randn(1) + 4)
        idx = int(torch.randint(1, int((ORIGINAL_DURATION-audio_duration)*self.sr/precision), (1,)))

        x[:,0:int(idx*precision)], x[:,int(idx*precision+audio_duration*self.sr):] = 0, 0

        return x

    def random_filters(self, x):
        """
        Return random filter
        """
        idx = torch.randint(0, len(self.filters)-1, (1,))
        xfilt = sig.lfilter(self.filters[idx], [1.], x.numpy())

        return torch.tensor(np.real(xfilt), dtype=torch.float)

    def create_white_noise(self, x):
        """
        Return white noise
        """
        snr = int(torch.randint(6,32, (1,)))
        wgn = torch.randn(int(self.audio_duration*self.sr))
        veff = torch.sqrt(torch.mean(x**2))
        wgn = wgn * veff * 10**(-snr/20)

        return wgn

    def __getitem__(self, index):
        file_name = os.path.basename(self.file_names[index])
        sound, sr = librosa.load(self.path + file_name, sr=None, mono=None)
        tensor_sound = torch.from_numpy(sound).float()

        if self.mode == 'test':
            tensor_sound = tensor_sound.to(DEVICE)
            return tensor_sound, self.classe[file_name.split('-')[0]]

        elif self.mode == 'train':

            if int(torch.randint(0,2,(1,))) == 1 and self.transform:
                tensor_sound = self.random_crop(tensor_sound)

            if int(torch.randint(0,2,(1,))) == 1 and self.transform:
                tensor_sound = self.random_filters(tensor_sound)

            if int(torch.randint(0,2,(1,))) == 1 and self.transform:
                noise = self.create_white_noise(tensor_sound)
                tensor_sound = tensor_sound + noise

            tensor_sound = tensor_sound.to(DEVICE)

            return tensor_sound, self.classe[self.label[index]]

        elif self.mode == 'eval':
            return tensor_sound

    def __len__(self):
        return len(self.file_names)


def split_train_val_df(original_train_path, train_path, valid_path, random_state):
    """
    If train/val datasets do not exist, create them by splitting the original train set.
    Else load the train/val datasets.
    """
    if not os.path.isfile(valid_path) or not os.path.isfile(train_path):
        original_train_df = pd.read_csv(original_train_path, sep='\t')

        train_df, val_df = train_test_split(original_train_df, test_size=0.2, 
                                            random_state=random_state, stratify=original_train_df[['scene_label']])

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # Save to csv
        train_df.to_csv(train_path, sep='\t')
        val_df.to_csv(valid_path, sep='\t')

        print("Save new train and val datasets")

    else:
        train_df = pd.read_csv(train_path, sep='\t')
        val_df = pd.read_csv(valid_path, sep='\t')

        print("Load existing train and val datasets")

    return (train_df, val_df)


def create_dataset(batch_size, frac_data=1, random_state=17, data_augment=''):
    """
    Create dataloaders and save them at {DATA_PATH}.
    """
    test_df = pd.read_csv(TEST_LABELS_PATH)

    train_df, val_df = split_train_val_df(original_train_path=OLD_TRAIN_LABELS_PATH, 
                                          train_path=TRAIN_LABELS_PATH, valid_path=VAL_LABELS_PATH, 
                                          random_state=random_state)

    # Take {frac_data}% of the dataset for quick test.
    if frac_data < 1:
        train_df = train_df.sample(frac=frac_data, random_state=random_state).reset_index(drop=True)
        val_df = val_df.sample(frac=frac_data, random_state=random_state).reset_index(drop=True)
        test_df = test_df.sample(frac=frac_data, random_state=random_state).reset_index(drop=True)


    # Load audios
    print("Load audios...")
    train_set = DCASEDataset(train_df['filename'],train_df['scene_label'], PATH, transform=data_augment)
    val_set = DCASEDataset(val_df['filename'],val_df['scene_label'], PATH, transform=data_augment)
    test_set = DCASEDataset(test_df['filename'], None, PATH, mode='test')

    # Create dataloaders
    print("Create dataloaders...")
    if MULTIPROCESS:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = NUM_CORE-1)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers = NUM_CORE-1)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    torch.save(train_loader,f'{DATA_PATH}train_data.pt')
    torch.save(val_loader,f'{DATA_PATH}val_data.pt')
    torch.save(test_loader,f'{DATA_PATH}test_data.pt')

    print(f"All datasets saved at {DATA_PATH}")

    dataloaders_length = [len(train_set), len(val_set), len(test_set)]

    return dataloaders_length


def create_eval_dataset(eval_df, eval_data_path, frac_data=1, random_state=17):
    """
    Create dataloaders and save them at {DATA_PATH}.
    """

    # Take {frac_data}% of the dataset for quick test.
    if frac_data < 1:
        eval_df = eval_df.sample(frac=frac_data, random_state=random_state).reset_index(drop=True)

    # Load audios
    print("Load audios...")
    eval_set = DCASEDataset(eval_df['filename'], None, eval_data_path, mode='eval')

    # Create dataloaders
    print("Create dataloaders...")
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False)

    torch.save(eval_loader,f'{DATA_PATH}evaluation_data.pt')

    print(f"Evaluation dataset saved at {DATA_PATH}")

    evalloader_length = len(eval_set)

    return evalloader_length



if __name__ == '__main__':

    BATCH_SIZE = 64
    FRAC_DATA = 1                     # Take {FRAC_DATA}% of the dataset
    DATA_AUGMENT = 1                  # int 0: no, 1: yes
    RANDOM_STATE = 17
    random.seed(RANDOM_STATE)

    dataloaders_length = create_dataset(batch_size=BATCH_SIZE, frac_data=FRAC_DATA, 
                                        random_state=RANDOM_STATE, data_augment=DATA_AUGMENT)

    trainloader = torch.load(f'{DATA_PATH}train_data.pt')
    validationloader = torch.load(f'{DATA_PATH}val_data.pt')
    testloader = torch.load(f'{DATA_PATH}test_data.pt')

    dataloaders = {"train": trainloader,
                   "val": validationloader,
                   "test": testloader}

    dataset_sizes = {"train": dataloaders_length[0],
                     "val": dataloaders_length[1],
                     "test": dataloaders_length[2]}

    print(dataset_sizes)
