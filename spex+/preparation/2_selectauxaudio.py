import shutil
import numpy as np
import os
import logging
import json

exp_workspace = "/Work21/2021/fuyanjie/pycode/libri_seg_spex+/data"
dataset_path = "/Work21/2020/lijunjie/USSfromjingru/Junjie_Simulation/Libirmix/data/mixture/Librispeech_mixture"
Libri_path = "/CDShare3/LibriSpeech_wav8k"

split2librisplit = {
    "train" : "train-clean-100",
    "dev" : "dev-clean",
    "test" : "test-clean"
    }

def uttid2aux_audio(split):
    json_path = os.path.join(exp_workspace, 'spkid2uttidpaths', f'{split2librisplit[split]}.json')

    f = open(json_path, 'r')
    spkid2uttidpaths = json.loads(f.read())
    mix_path = os.path.join(dataset_path, split, 'mix_2sources')
    s1_path = os.path.join(dataset_path, split, 'clean_2sources', 's1')
    aux_path = os.path.join(dataset_path, split, 'aux_s1')
    os.makedirs(aux_path, exist_ok=True)
    for root, dirs, files in os.walk(mix_path):
        files.sort()
        for idx, filename in enumerate(files):
            random_state = np.random.RandomState(idx)
            uttid = filename[:-4]
            s1spkid = uttid.split('-')[0]
            s1chapid = uttid.split('-')[1]
            s1uttid = uttid.split('-')[2]
            s1uttid = f'{s1spkid}-{s1chapid}-{s1uttid}'
            uttidpaths = spkid2uttidpaths[s1spkid]
            spk1uttids = [uttid for uttid in uttidpaths.keys()]

            spk1uttids.remove(s1uttid)

            rand_idx = random_state.randint(0, len(spk1uttids)-1)
            aux_uttid = spk1uttids[rand_idx]
            # rename mixture and s1 filename (add f'_{aux_uttid}')
            # os.rename(os.path.join(root, filename), os.path.join(root, uttid+f'_{aux_uttid}.wav'))
            # os.rename(os.path.join(s1_path, filename), os.path.join(s1_path, uttid+f'_{aux_uttid}.wav'))
            aux_src_path = uttidpaths[aux_uttid]
            aux_dst_path = os.path.join(aux_path, uttid+'.wav')
            print("aux_src_path: ", aux_src_path)
            print("aux_dst_path: ", aux_dst_path)
            shutil.copy(aux_src_path, aux_dst_path)
    return 

def main():
    logging.basicConfig(level=logging.INFO)
    uttid2aux_audio("dev")
    logging.info("Finished creating cv aux audios")
    uttid2aux_audio("test")
    logging.info("Finished creating test aux audios")
    uttid2aux_audio("train")
    logging.info("Finished creating train aux audios")


if __name__ == '__main__':
    main()
