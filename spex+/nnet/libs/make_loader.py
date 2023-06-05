import sys
sys.path.append('/Work21/2021/fuyanjie/pycode/wsj0_seg_spex+_baseline')
import random
import torch
import numpy as np
import pandas
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nnet.libs.audio import WaveReader
from nnet.conf import trainer_conf, nnet_conf, train_data, dev_data, chunk_size

np.set_printoptions(threshold=np.inf)

class MyDataset(Dataset):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp="", ref_scp=None, aux_scp=None, metadata_path=None, spk_list=None, sample_rate=8000):
        super(MyDataset, self).__init__()
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = WaveReader(ref_scp, sample_rate=sample_rate)
        self.aux = WaveReader(aux_scp, sample_rate=sample_rate)
        # If use WSJ0-2mix data (min version), don't need this part
        # self.ref_dur = WaveReader("data/uniq_target_ref_dur.txt", sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.spk_list = self._load_spk(spk_list)
        print(self.spk_list)
        self.metadata = pandas.read_csv(metadata_path, index_col=0)
        # print(self.metadata)


    def _load_spk(self, spk_list_path):
        print(f'spk_list_path {spk_list_path}')
        if spk_list_path is None:
            return []
        lines = open(spk_list_path).readlines()
        new_lines = []
        for line in lines:
            new_lines.append(line.strip())

        return new_lines

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        # B
        key = self.mix.index_keys[index] 
        mix = self.mix[key]
        ref = self.ref[key]
        aux = self.aux[key]
        metadata_one_audio = self.metadata.loc[key]

        source_1_start = metadata_one_audio.at["source_1_start"]
        source_1_end = metadata_one_audio.at["source_1_end"]
        source_2_start = metadata_one_audio.at["source_2_start"]
        source_2_end = metadata_one_audio.at["source_2_end"]
        audio_len = metadata_one_audio.at["audio_length"]

        # spk_idx = self.spk_list.index(key.split('_')[0][0:3])

        return {
            "key": key,
            "mix": mix[:audio_len].astype(np.float32),
            "ref": ref[:audio_len].astype(np.float32),
            "aux": aux.astype(np.float32),
            "aux_len": len(aux),
            "source_1_start": source_1_start,
            "source_1_end": source_1_end,
            "source_2_start": source_2_start,
            "source_2_end": source_2_end,
            "audio_len": audio_len,
            # "spk_idx": spk_idx
        }

def my_collate_fn(dataBatch):
    # len(dataBatch): batchsize
    key = []
    mix = []
    ref = []
    aux = []
    aux_len = []
    source_1_start = []
    source_1_end = []
    source_2_start = []
    source_2_end = []
    audio_len = []

    for idx, eg in enumerate(dataBatch):
        key.append(eg["key"])
        mix.append(eg["mix"])
        ref.append(eg["ref"])
        aux.append(eg["aux"])
        aux_len.append(eg["aux_len"])
        source_1_start.append(eg["source_1_start"])
        source_1_end.append(eg["source_1_end"])
        source_2_start.append(eg["source_2_start"])
        source_2_end.append(eg["source_2_end"])
        audio_len.append(eg["audio_len"])
    
    audio_max_len = np.max(audio_len)

    for idx in range(len(key)):
        mix_idx = mix[idx]
        padding_data = np.pad(mix_idx, (0, audio_max_len-audio_len[idx]), 'constant', constant_values=0.0)
        mix[idx] = padding_data

        ref_idx = ref[idx]
        padding_data = np.pad(ref_idx, (0, audio_max_len-audio_len[idx]), 'constant', constant_values=0.0)
        ref[idx] = padding_data

    mix_np = np.stack(mix) # [B, T] mix.type: numpy.ndarray
    ref_np = np.stack(ref) # [B, T] ref.type: numpy.ndarray (same T with mix, i.e., audio_max_len)

    aux_max_len = np.max(aux_len)

    for idx in range(len(key)):
        aux_idx = aux[idx]
        padding_data = np.pad(aux_idx, (0, aux_max_len-aux_len[idx]), 'constant', constant_values=0.0)
        aux[idx] = padding_data

    aux_np = np.stack(aux) # [B, T_aux, i.e., aux_max_len not equal to audio_max_len] mix.type: numpy.ndarray

    mix = torch.from_numpy(mix_np)
    ref = torch.from_numpy(ref_np)
    aux = torch.from_numpy(aux_np)


    return {
        'key': key,
        'mix': mix,
        'ref': ref,
        'aux': aux,
        'aux_len': torch.tensor(aux_len, dtype=torch.int),
        'source_1_start': source_1_start,
        'source_1_end': source_1_end,
        'source_2_start': source_2_start,
        'source_2_end': source_2_end,
        'audio_max_len': audio_max_len
    }


def make_loader(train_or_dev='train', data_kwargs=None, batch_size=4, num_workers=8, pin_memory=False):
    dataset = MyDataset(**data_kwargs)

    if train_or_dev == 'train':
        shuffle = True
    else:
        shuffle = False
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=True, collate_fn=my_collate_fn,
                            pin_memory=pin_memory)
    return dataloader


if __name__ == "__main__":
    data_loader = make_loader(dev_data, 'dev')
    print(f'len(data_loader) {len(data_loader)}') # len(data_loader) is samples / batch_size
    one_batch_data = next(iter(data_loader)) 
    print('AAA ', one_batch_data["key"])
    print('BBB ', one_batch_data["key"])
    print('CCC ', one_batch_data["key"])
    print('DDD ', one_batch_data["key"])
    # AAA  torch.Size([224, 12, 7, 257]) torch.Size([224])
    # one_batch_data = next(iter(data_loader))
    # print('BBB ', one_batch_data["clean_ris"].shape, one_batch_data["target_doa"].shape)
    for idx,data in enumerate(data_loader):
        # sf.write('./'+str(idx)+'_mix.wav',data['mix_audio'].numpy().squeeze(),samplerate=8000)
        # sf.write('./sample/'+str(idx)+'_mix.wav',data['audio_data'].numpy().squeeze(),samplerate=8000)
        # sf.write('./sample/'+str(idx)+'_mix.wav',data['audio_data'].unsqueeze(),samplerate=8000)
        # print('ff', data['video_data'].shape)  # (b,1,75,70,90)
        # # print('ff', data['mix_audio'].numpy().shape)  # (b,1,24000)
        # # print(data['target_audio'].numpy().shape)
        # image = data['video_data'][0,0,0]
        # image = image.unsqueeze(-1)
        # print(image.shape)
    
        # print(data['uttid'])
        # plt.imshow(image,cmap='gray')
        # plt.savefig('./x.png')

        # print('ff', data['feature_mask'].shape)  # (b,1,24000)
        # sdr,_,_,_ = bss_eval_sources(data['target_audio'].numpy()[0],data['mix_audio'].numpy()[0])
        # print(sdr)
        # print('ff', data['feature'].shape)
        # print(data['feature_length'])
        # print(data['audio_length'])
        print(data['key'])
        print(data['mix'].shape)
        print(data['ref'].shape)
        print(data['aux'].shape)
        print(data['aux_len'])
        print(data['source_1_start'])
        print(data['source_2_end'])
        print(data['source_1_start'])
        print(data['source_2_end'])
        print(data['audio_len'])

        # print('------------')
        if idx >= 5:
            break