'''
DataLoader for augmentation
'''
import logging
log = logging.getLogger()
from glob import glob
import numpy
import os
import random
import soundfile
import torch
from scipy import signal

# musan_path /datasets1/musan_split
# rir_path   /datasets1/RIRS_NOISES/simulated_rirs


class dataAugemntation(object):
    def __init__(self, musan_path, mode):
        self.mode = mode
        self.max_length = 32000  # 这里需要指定最大长度是多少。是4s*16000=64000
        # Load and configure augmentation files
        # musan_path = '/CDShare/musan'
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 20], 'speech': [0, 20], 'music': [0, 20]}
        self.numnoise = {'noise': [1, 1], 'speech': [1, 1], 'music': [1, 1]}
        self.noiselist = {}
        # augment_files = glob(os.path.join(musan_path, '*/*/*.wav'), recursive=True)
        augment_files = glob(os.path.join(musan_path, '**/*.wav'), recursive=True)
        # log.info(len(augment_files))
        t = 0
        if mode == 'test':
            t=1
        for i, (file) in enumerate(augment_files):
            # log.info(i)
            if i % 2 == t:
                # log.info(file)
                if file.split('/')[-3] not in self.noiselist:
                    self.noiselist[file.split('/')[-3]] = []
                self.noiselist[file.split('/')[-3]].append(file)

    def getitem(self, audio, augtype, snr=20):
        # Read the utterance and randomly select the segment
        # audio, sr = soundfile.read(filename)
        length = self.max_length
        if self.mode == 'test':
            length = audio.shape[0]
            self.noisesnr = {'noise': [snr, snr], 'speech': [snr, snr], 'music': [snr, snr]}
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        # Data Augmentation
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1: # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 2: # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 3:
            audio = self.add_noise(audio, 'speech')
        elif augtype == 4: # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 5: # Noise & Reverberation
            audio = self.add_noise(audio, 'noise')
            audio = self.add_rev(audio)
        elif augtype == 6: # white gussian noise
            snr = random.randint(self.numnoise['noise'][0],self.numnoise['noise'][1])
            audio = self.wgn(audio, snr)
        elif augtype == 7: # Fade in & out
            audio = self.fade(torch.from_numpy(audio.reshape(1,-1))).numpy()
        return audio[0]

    def add_rev(self, audio):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.max_length + 240]

    def add_noise(self, audio, noisecat):
        clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        length = self.max_length
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            if self.mode == 'test':
                length = audio.shape[1]
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio],axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio