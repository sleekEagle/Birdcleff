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
from omegaconf import DictConfig, OmegaConf
import hydra
from PIL import Image

@hydra.main(version_base=None, config_path="conf", config_name="config")
def save_spec(cfg):
    data_df = pd.read_csv(cfg.TRAIN_CSV_PATH)

    for idx in range(len(data_df)):
        print(f'Processing data {idx} of {len(data_df)}',end='\r')
        data_row = data_df.iloc[idx]
        path = os.path.join(cfg.TRAIN_AUDIO_PATH, data_row.filename)
        audio_data, _ = librosa.load(path, sr=cfg.FS)
        n_samples = int(cfg.FS * cfg.TARGET_DURATION)
        n_seg = math.ceil(audio_data.shape[0]/(n_samples))
        pad_len = int(n_seg * cfg.FS * cfg.TARGET_DURATION - audio_data.shape[0])
        if pad_len > 0:
            audio_data = np.pad(audio_data, (0, pad_len), mode='constant')
        else:
            audio_data = audio_data[:n_seg * n_samples]
        # data = audio_data.reshape(-1, n_seg)
        
        #normalize
        audio_data = (audio_data - np.mean(audio_data)) / np.std(audio_data)
        s_cfg = cfg.spectrogram
        stft = librosa.stft(audio_data, n_fft=s_cfg.linear.n_fft,
                    hop_length=s_cfg.linear.hop_length,
                    win_length=s_cfg.linear.n_fft)
        S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        S_normalized = ((S_db - S_db.min()) * (255 / (S_db.max() - S_db.min()))).astype(np.uint8)
        out_path = os.path.join(cfg.spectrogram.output_path, data_row.filename.replace('.ogg', '.png'))
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        Image.fromarray(S_normalized).save(out_path)

        # import matplotlib.pyplot as plt
        # librosa.display.specshow(S_db, 
        #                 sr=cfg.FS,  # you'll need to provide your original sample rate
        #                 hop_length=s_cfg.linear.hop_length,
        #                 x_axis='time', 
        #                 y_axis='log')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Spectrogram')

if __name__ == "__main__":
    save_spec()



'''
Problamatic audios:
    1139490
'''