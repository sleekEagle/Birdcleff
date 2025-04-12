import os
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional
import json
import pandas as pd
import math
from torchvision import transforms
import torch

def explore_audio_directory(directory):
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ogg') or file.endswith('.mp3') or file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def get_audio_duration(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=10)  # Charger seulement les 10 premières secondes pour être rapide
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path}: {e}")
        return None

def calc_duration_stats(data_path):
    train_audio_files = explore_audio_directory(data_path)
    print(f"Nombre de fichiers audio d'entraînement: {len(train_audio_files)}")
    print(f"Exemples de fichiers audio: {[os.path.basename(f) for f in train_audio_files[:5]]}...")

    durations = []
    for i,file in enumerate(train_audio_files):
        print(f'{i} or {len(train_audio_files)} is Processed',end='\r')
        duration = get_audio_duration(file)
        if duration is not None:
            durations.append(duration)

class BirdDataset(Dataset):
    def __init__(self, cfg, mode='TRAIN', split='fold', n_split=1):
        
        self.cfg=cfg
        self.n_split = n_split
        self.mode = mode
        df = pd.read_csv(cfg.TRAIN_CSV_PATH)
        df['label'], unique_categories = pd.factorize(df['primary_label'])
        self.num_classes = len(unique_categories)
        #read splits file
        with open(cfg.SPLIT_FILE, "r") as json_file:
            self.splits = json.load(json_file)
        idx = self.splits[str(n_split)][mode]
        self.data_df = df.iloc[idx]
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((256,256))          # Resize shorter side to 256 (aspect ratio maintained)
                ])

        # train_audio_files = explore_audio_directory(cfg.TRAIN_AUDIO_PATH)
        # print(f"Number of audio files: {len(train_audio_files)}")

    def __len__(self):
        return len(self.data_df)
    
    def get_num_classses(self):
        return self.num_classes

    def check_str_in_list(self,f, lst):
        return any(x in f for x in lst)
    
    def __getitem__(self, idx):
        data_row = self.data_df.iloc[idx]
        path = os.path.join(self.cfg.TRAIN_AUDIO_PATH, data_row.filename)
        audio_data, _ = librosa.load(path, sr=self.cfg.FS)
        n_samples = int(self.cfg.FS * self.cfg.TARGET_DURATION)
        n_seg = math.ceil(audio_data.shape[0]/(n_samples))
        pad_len = int(n_seg * self.cfg.FS * self.cfg.TARGET_DURATION - audio_data.shape[0])
        if pad_len > 0:
            audio_data = np.pad(audio_data, (0, pad_len), mode='constant')
        else:
            audio_data = audio_data[:n_seg * n_samples]

        if self.mode == 'TRAIN':
            #select a segment in random
            start = np.random.randint(0, n_seg) * n_samples
            data = audio_data[start:start+n_samples]
            data = np.expand_dims(data, axis=1)
        elif self.mode == 'VAL':
            data = audio_data.reshape(-1, n_seg)

        #normalize
        data = (data - np.mean(data)) / np.std(data)
        data = np.swapaxes(data, 0, 1)
        s_cfg = self.cfg.spectrogram
        spec = torch.empty(0)
        if s_cfg.type == 'linear':
            for d in data:
                stft = librosa.stft(d, n_fft=s_cfg.linear.n_fft,
                                    hop_length=s_cfg.linear.hop_length,
                                    win_length=s_cfg.linear.n_fft)
                magnitude_spectrogram = np.abs(stft)
                power_spectrogram = magnitude_spectrogram ** 2
                spec = torch.concat([spec,self.transform(power_spectrogram)],dim=0)

            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(power_spectrogram, 
            #                         sr=self.cfg.FS, 
            #                         hop_length=s_cfg.linear.hop_length, 
            #                         x_axis='time', 
            #                         y_axis='linear',
            #                         cmap='viridis')
            # plt.colorbar(label='Magnitude')
            # plt.title('Linear Spectrogram (Magnitude)')
            # plt.show()
            
        label = data_row['label']
        label_oh = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes)
        
        data = {}
        data['spec'] = spec
        data['label'] = label_oh
        return data
    
class BirdModule(LightningDataModule):
    def __init__(self,cfg,n_split=1):
        super().__init__()
        self.cfg=cfg
        self.n_split = n_split
        self.num_classes = -1

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset=BirdDataset(self.cfg,mode='TRAIN',
                                        split=self.cfg.splits.type,
                                        n_split=self.n_split)
        self.num_classes = self.train_dataset.get_num_classses()
        self.val_dataset=BirdDataset(self.cfg,mode='VAL',
                                      split=self.cfg.splits.type,
                                      n_split=self.n_split)
        
    def get_num_classes(self):
        return self.num_classes
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                            batch_size=self.cfg.bs,
                            num_workers=self.cfg.num_workers)

    def val_dataloader(self) -> DataLoader:
        print(f'Val Batch size : {self.cfg.bs}')
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, 
        num_workers=self.cfg.num_workers,
        persistent_workers=False,
        pin_memory=True)
    
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def print_stats(cfg : DictConfig) -> None:
    ttsDM=BirdModule(cfg, n_split=1)
    ttsDM.setup()
    all_data = ttsDM.all_dataloader()
    calc_duration_stats()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def create_splits(cfg : DictConfig) -> None:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cfg.splits.n_folds, shuffle=True, random_state=cfg.seed)

    # taxonomy_df = pd.read_csv(cfg.TAXONOMY_PATH)
    # species_ids = taxonomy_df['primary_label'].tolist()
    splits = {}
    df = pd.read_csv(cfg.TRAIN_CSV_PATH)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        splits[fold] = {
            'TRAIN': train_idx.tolist(),
            'VAL': val_idx.tolist()
        }
    with open(cfg.SPLIT_FILE, "w") as json_file:
        json.dump(splits, json_file, indent=4)

if __name__ == "__main__":
    print_stats()
