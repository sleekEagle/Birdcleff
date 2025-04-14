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
    def __init__(self, cfg, mode='TRAIN', split='fold', n_split=1, data_dict=None,num_classes=1):
        
        self.cfg=cfg
        self.n_split = n_split
        self.mode = mode
        self.data_dict = data_dict
        self.num_classes = num_classes

        # read splits file
        with open(cfg.SPLIT_FILE, "r") as json_file:
            self.splits = json.load(json_file)
        self.valid_indices = self.splits[str(n_split)][mode]

    def __len__(self):
        return len(self.valid_indices)

    def check_str_in_list(self,f, lst):
        return any(x in f for x in lst)
    
    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        label = self.data_dict[i]['label']
        spec = self.data_dict[i]['spec']
        #select one spectrogram in random for training
        if self.mode == 'TRAIN':
            random_int = np.random.randint(0, spec.size(0))
            spec = spec[random_int].unsqueeze(0)

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
        data_dict, self.num_classes = read_all_data(self.cfg)
        self.train_dataset=BirdDataset(self.cfg,mode='TRAIN',
                                        split=self.cfg.splits.type,
                                        n_split=self.n_split,
                                        data_dict=data_dict,
                                        num_classes=self.num_classes)
        self.val_dataset=BirdDataset(self.cfg,mode='VAL',
                                      split=self.cfg.splits.type,
                                      n_split=self.n_split,
                                      data_dict=data_dict,
                                      num_classes=self.num_classes)
        
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
    read_all_data()
