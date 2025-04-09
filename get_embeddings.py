'''
Exploring the clustering cababilities of the perch model
'''
from pathlib import Path
import glob
import librosa
import re
import tensorflow_hub as hub
import os
import numpy as np

TRAIN_DATA_PATH_LABELLED = Path('D:\\birdclef-2025\\train_audio')
TRAIN_DATA_PATH_UNLABELLED = Path('D:\\birdclef-2025\\train_soundscapes')
SR = 32_000
OUTPUT_PATH = Path('D:\\birdclef-2025\\tr_emb_unlb')

birdvoc_model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8')


def get_files(path):
    ogg_files = list(path.rglob("*.ogg"))
    print(f'Num files: {len(ogg_files)}')
    return ogg_files

def read_data(path,sr=SR):
    data, _ = librosa.load(path, sr=sr)
    data_norm = 0.25 * (data - data.min())/(data.max() - data.min())
    return data_norm

oggs = get_files(TRAIN_DATA_PATH_UNLABELLED)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
               
all_embeddings = {}
    
for i,file in enumerate(oggs):
    print(f'{i} of {len(oggs)} processing...',end='\r')
    f = os.path.basename(file).split('.')[0]
    out_f_name = os.path.join(OUTPUT_PATH,f+'.npy')
    if os.path.exists(out_f_name):
        continue
    data = read_data(file,sr=SR)

    model_outputs = birdvoc_model.infer_tf(data.reshape((-1, 5 * SR)))
    emb = model_outputs['embedding']
    # all_embeddings[fname] = emb
    np.save(out_f_name,emb)