from random import sample
import soundfile as sf
from python_speech_features import mfcc, fbank, logfbank, delta
from librosa import stft, magphase
from torch.utils import data
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import os
import torch.nn.functional as F
# import torchaudio
import torch.nn as nn
# from torch.utils.data import DataLoader
from generator.augmentation import dataAugemntation
# from generator.noise92 import dataAugemntation
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
log = logging.getLogger()
class SpkTrainDataset(Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.TRAIN_MANIFEAT = opts["train_manifest"]
        self.feat_type = opts['feat_type']
        self.rate = opts['rate']
        self.data_feat, self.data_label, self.speaker_dict = self._load_dataset()
        self.count = len(self.data_feat)
        # self.augmentation = dataAugemntation('/CDshare/musan')
        self.augmentation = dataAugemntation('/CDShare/musan', 'train')

    def _load_dataset(self):
        with open(self.TRAIN_MANIFEAT, 'r') as f:
            all_data = f.read()
        all_data = all_data.split('\n')

        data_feat = []
        data_label = []
        speaker_dict = {}
        speaker_id = 0
        for data in all_data:
            if data == '':
                continue
            speaker, path = data.split(' ')[0], data.split(' ')[1]

            if speaker not in speaker_dict.keys():
                speaker_dict[speaker] = speaker_id
                speaker_id += 1

            data_label.append(speaker_dict[speaker])
            data_feat.append(path)
        
        # data_feat = np.array(data_feat)
        data_label = np.array(data_label)
        log.info('Found {} different speakers.'.format(str(speaker_id).zfill(5)))

        return data_feat, data_label, speaker_dict

    def _fix_length(self, feat):
        max_length = self.opts['max_length']
        out_feat = feat
        while out_feat.shape[0] < max_length:
            out_feat = np.concatenate((out_feat, feat), axis=0)
        feat_len = out_feat.shape[0]
        start = random.randint(a=0, b=feat_len-max_length)
        end = start + max_length
        out_feat = out_feat[start:end,]
        return out_feat

    def _load_audio(self, path):
        y, sr = sf.read(path)
        return y, sr

    def _normalize(self, feat):
        return (feat - feat.mean(axis = 0)) / (feat.std(axis = 0) + 2e-12)

    def _delta(self, feat, order = 2):
        if order == 2:
            feat_d1 = delta(feat, N = 1)
            feat_d2 = delta(feat, N = 2)
            feat = np.hstack([feat, feat_d1, feat_d2])
        elif order == 1:
            feat_d1 = delta(feat, N = 1)
            feat = np.hstack([feat, feat_d1])
        return feat

    def _extract_feature(self, data):
        if self.feat_type == 'mfcc':
            feat = mfcc(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], numcep = self.opts['num_cep'])
        elif self.feat_type == 'fbank':
            feat, _ = fbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'logfbank':
            feat = logfbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'stft':
            feat = stft(data, n_fft = self.opts['n_fft'], hop_length = int(self.rate * self.opts['win_shift']), win_length = int(self.rate * self.opts['win_len']))
            feat, _ = magphase(feat)
            feat = np.log1p(feat)
            feat = feat.transpose(1,0)
        else:
            raise NotImplementedError("Other features are not implemented!")
        if self.opts['normalize']:
            feat = self._normalize(feat)
        if self.opts['delta']:
            feat = self._delta(feat, order = 2)
        return feat

    def collate_fn(self, batch):
        clean_feats = []
        random_noises = []
        labels = []
        for id in batch:
            audio_path = self.data_feat[id]
            # audio_path = '/local01/liumeng/voxceleb2/audio/dev/aac/' + audio_path
            source_feat, _ = self._load_audio(audio_path)
            if source_feat.shape[0] == 0:
                continue
            source_feat = self._fix_length(source_feat)
            noise_type = random.randint(1,3)
            clean_feat = dataAugemntation.getitem(self.augmentation, source_feat, 0)
            random_noise = dataAugemntation.getitem(self.augmentation, source_feat, noise_type)
            
            clean_feat = self._extract_feature(clean_feat)
            clean_feat = clean_feat.astype(np.float32)
            clean_feat = np.array(clean_feat)
            clean_feats.append(clean_feat)

            random_noise = self._extract_feature(random_noise)
            random_noise = random_noise.astype(np.float32)
            random_noise = np.array(random_noise)
            random_noises.append(random_noise)

            label = self.data_label[id]
            labels.append(label)
        clean_feats = np.array(clean_feats).astype(np.float32)
        random_noises = np.array(random_noises).astype(np.float32)
        labels = np.array(labels).astype(np.int64)
        # return torch.from_numpy(clean_feats), torch.from_numpy(labels)
        return torch.from_numpy(clean_feats), torch.from_numpy(labels), torch.from_numpy(random_noises)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        idx = idx % self.count
        return idx




class SpkTestDataset(Dataset):
    def __init__(self, opts, augtype, noise_snr):
    # def __init__(self, opts):
        self.opts = opts
        self.TEST_MANIFEAT = opts["test_manifest"]
        self.feat_type = opts['feat_type']
        self.rate = opts['rate']
        self.augtype = augtype
        self.noise_snr = noise_snr
        self.augmentation = dataAugemntation('/CDShare/musan', 'test')
        # self.augmentation = dataAugemntation('/datasets1/musan_split', 'test')
        self.data_feat, self.data_label, self.speaker_dict = self._load_dataset()
        self.count = len(self.data_feat)

    def _load_dataset(self):
        with open(self.TEST_MANIFEAT, 'r') as f:
            all_data = f.read()
        all_data = all_data.split('\n')

        data_feat = []
        data_label = []
        speaker_dict = {}
        speaker_id = 0
        for data in all_data:
            if data == '':
                continue
            data = data.replace('.pkl', '.wav')
            speaker, path = data.split('/')[0], data
            # data_label.append(path)
            data_label.append(path)
            path = '/CDShare/voxceleb1/vox1_test_wav/' + path
            source_feat, _ = self._load_audio(path)
            source_feat = self.augmentation.getitem(source_feat, self.augtype, self.noise_snr)
            source_feat = self._extract_feature(source_feat)
            # path = '/CDShare/voxceleb1/vox1_test_wav/' + path

            if speaker not in speaker_dict.keys():
                speaker_dict[speaker] = speaker_id
                speaker_id += 1
            
        #     feat = feat.astype(np.float32)
        #     feat = np.array(feat)
            data_feat.append(source_feat)
        
        data_feat = np.array(data_feat)
        data_label = np.array(data_label)
        # log.info('Found {} different speakers.'.format(str(speaker_id).zfill(5)))

        return data_feat, data_label, speaker_dict


    def _load_audio(self, path):
        y, sr = sf.read(path)
        return y, sr

    def _normalize(self, feat):
        return (feat - feat.mean(axis = 0)) / (feat.std(axis = 0) + 2e-12)

    def _delta(self, feat, order = 2):
        if order == 2:
            feat_d1 = delta(feat, N = 1)
            feat_d2 = delta(feat, N = 2)
            feat = np.hstack([feat, feat_d1, feat_d2])
        elif order == 1:
            feat_d1 = delta(feat, N = 1)
            feat = np.hstack([feat, feat_d1])
        return feat
    
    def _extract_feature(self, data):
        if self.feat_type == 'mfcc':
            feat = mfcc(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], numcep = self.opts['num_cep'])
        elif self.feat_type == 'fbank':
            feat, _ = fbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'logfbank':
            feat = logfbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'stft':
            feat = stft(data, n_fft = self.opts['n_fft'], hop_length = int(self.rate * self.opts['win_shift']), win_length = int(self.rate * self.opts['win_len']))
            feat, _ = magphase(feat)
            feat = np.log1p(feat)
            feat = feat.transpose(1,0)
        else:
            raise NotImplementedError("Other features are not implemented!")
        if self.opts['normalize']:
            feat = self._normalize(feat)
        if self.opts['delta']:
            feat = self._delta(feat, order = 2)
        return feat

    def collate_fn(self, batch):
        feats = []
        labels = []
        for id in batch:
            feat = self.data_feat[id]
            label = self.data_label[id]
            feats.append(feat)
            labels.append(label)
        feats = np.array(feats).astype(np.float32)
        labels = np.array(labels)
        return torch.from_numpy(feats), labels


    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        idx = idx % self.count
        return idx