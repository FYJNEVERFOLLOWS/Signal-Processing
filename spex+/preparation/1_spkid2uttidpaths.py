import os
import argparse
import logging
import json

exp_workspace = "/Work21/2021/fuyanjie/pycode/libri_seg_spex+/data"

def generateSpkid2uttidpaths(split, dataPath):
    spkid2uttidpaths = dict()
    spkid2paths = dict()
    spkids = os.listdir(os.path.join(dataPath, split))
    for spkid in spkids:
        chapids = os.listdir(os.path.join(dataPath, split, spkid))
        for chapid in chapids:
            chap_folder = os.path.join(dataPath, split, spkid, chapid)
            audio_fnames = os.listdir(chap_folder)
            for audio_fname in audio_fnames:
                audio_path = os.path.join(chap_folder, audio_fname)
                print(f'audio_path {audio_path}')
                uttid = audio_fname.split('.')[0]
                spkid = uttid.split('-')[0]
                if not spkid in spkid2paths:
                    spkid2paths[spkid] = []
                spkid2paths[spkid].append(audio_path)
        
    for spkid in spkid2paths:
        paths = spkid2paths[spkid]
        uttidpaths = dict()
        for path in paths:
            uttid = path.split('/')[-1].split('.')[0]
            uttidpaths[uttid] = path
        spkid2uttidpaths[spkid] = uttidpaths

    os.makedirs(os.path.join(exp_workspace, 'spkid2uttidpaths'), exist_ok=True)
    json_path = os.path.join(exp_workspace, 'spkid2uttidpaths', f'{split}.json')
    json_str = json.dumps(spkid2uttidpaths)
    f = open(json_path, 'w')
    f.write(json_str)
    f.close()

def main(args):
    logging.basicConfig(level=logging.INFO)
    dataPath = args.data_dir
    # generateSpkid2uttidpaths("test-clean", dataPath)
    logging.info("Finished creating test")
    # generateSpkid2uttidpaths("dev-clean", dataPath)
    logging.info("Finished creating cv")
    generateSpkid2uttidpaths("train-clean-100", dataPath)
    logging.info("Finished creating train")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to create mixtures scp in Kaldi wav.scp format'
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help='Input audio path'
    )

    # python 1_spkid2uttidpaths.py --data_dir "/CDShare3/LibriSpeech_wav8k"

    args = parser.parse_args()
    main(args)
