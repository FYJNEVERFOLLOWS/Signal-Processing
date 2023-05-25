import os
import torch
import torch.utils.data as utils
from pathlib import Path
from utility_functions import audio_image_csv_to_dict, load_image
import numpy as np
import pickle

class CustomAudioVisualDataset(utils.Dataset):
    def __init__(self, audio_path, image_path=None, image_audio_csv_path=None, transform_image=None):
        self.audio_path = audio_path
        self.paths = []
        # self.audio_predictors_path = audio_predictors[1]
        self.image_path = image_path
        if image_path:
            print("AUDIOVISUAL ON")
            self.image_audio_dict = audio_image_csv_to_dict(image_audio_csv_path)
            self.transform = transform_image
        else:
            print("AUDIOVISUAL OFF")
        with open(self.audio_path, 'rb') as f:
            self.paths = pickle.load(f)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_pred_path = self.paths[idx]
        parent_path = Path(self.audio_path).parent.absolute()
        with open(os.path.join(parent_path, audio_pred_path+'.pkl'), 'rb') as f:
            audio_pred, _, audio_trg = pickle.load(f) #_代表path
            audio_pred = np.array(audio_pred) 
        # convert to tensor
        audio_pred = torch.tensor(audio_pred).float()
        audio_trg = torch.tensor(audio_trg).float()
         
        if self.image_path:
            image_name = self.image_audio_dict[audio_pred_path]
            img = load_image(os.path.join(self.image_path, image_name))

            if self.transform:
                img = self.transform(img)

            return (audio_pred, img), audio_trg
        
        return audio_pred, audio_trg

    
if __name__ == "__main__":
    data_path = "/Work21/2021/wanghonglong/datasets/L3DAS23_sepprocessed_100/test/task1_test_path.pkl"
    dataset = CustomAudioVisualDataset(data_path)
    print(dataset[0])