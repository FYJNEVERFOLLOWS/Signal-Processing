from importlib.util import spec_from_file_location
import numpy as np
import pandas as pd
import random
import torch
# import torchvision.transforms as transforms
from generator.SR_Dataset import TruncatedInputfromMFB, ToTensorInput
from Log import log
from torch.utils.data import Dataset
import math
import logging
log = logging.getLogger()

class baseGenerator(Dataset):
    '''
        opts: Options for database file location
        args: The number of selected classes, the number of samples per class
    '''
    def __init__(self, opts, args, file_loader):
        super(baseGenerator, self).__init__()

        self.opts = opts
        self.args = args
        self.data, self.n_classes, self.n_data = self._load_data()
        self.file_loader = file_loader
    
    def _load_data(self):
        '''
            ex: 0: [1.pkl, 2.pkl ...]
        '''
        # Loading data
        cn_DB = pd.DataFrame()

        if self.args.dataset != '':
            log.info(self.opts[self.args.dataset+'_database'])
            DB = pd.read_json(self.opts[self.args.dataset+'_database'])
        else:
            DB = cn_DB
        speaker_list = sorted(set(DB['speaker_id']))
        audio_list = DB['filename']
        spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
        DB['labels'] = DB['speaker_id'].apply(lambda x: spk_to_idx[x])
        num_speakers = len(speaker_list)
        num_utterances = len(DB)
        log.info('Found {} different speakers.'.format(str(num_speakers).zfill(5)))
        data = []
        for index, row in DB.iterrows():
            tmp = []
            tmp.append(row['labels'])
            tmp.append(row['filename'])
            # tmp[row['filename']] = row['labels']
            data.append(tmp)


        # data = {row['filename']: row['labels'] for index, row in DB.iterrows()} # <id, array(path)>

        return data, num_speakers, num_utterances

    def collate_fn(self, batch):
        feats = []
        labels = []
        noisy_feats = []
        for id in batch:
            _imgs = self.data[id][1]
            feature, noisy_feature = self.file_loader(_imgs)
            feats.append(feature)
            noisy_feats.append(noisy_feature)
            labels.append(self.data[id][0])
        # feature = feats + noisy_feats
        feats = np.array(feats).astype(np.float32)
        labels = np.array(labels).astype(np.int64)
        noisy_feats = np.array(noisy_feats).astype(np.float32)
        # feature = np.array(feature).astype(np.float32)

        return torch.from_numpy(feats), torch.from_numpy(labels), torch.from_numpy(noisy_feats)
        # return torch.from_numpy(feature), torch.from_numpy(labels)

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        # idx = idx % self.n_classes
        idx = idx % self.n_data
        return idx
