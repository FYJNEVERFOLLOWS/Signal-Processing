import os
import argparse
import logging

exp_workspace = "/Work21/2021/fuyanjie/pycode/libri_seg_spex+/data"

def genrateScp(split, dataPath):
    mix = os.path.join(dataPath, split, 'mix_2sources')
    aux = os.path.join(dataPath, split, 'aux_s1')
    ref = os.path.join(dataPath, split, 'clean_2sources', 's1')

    wav_scp_dir = os.path.join(exp_workspace, 'libri_seg_extr')
    os.makedirs(os.path.join(wav_scp_dir, split), exist_ok=True)

    mix_scp = os.path.join(wav_scp_dir, split, 'mix.scp')
    aux_scp = os.path.join(wav_scp_dir, split, 'aux.scp')
    ref_scp = os.path.join(wav_scp_dir, split, 'ref.scp')

    file_mix = open(mix_scp, 'w')
    for root, dirs, files in os.walk(mix):
        files.sort()
        for f in files:
            file_mix.write(f[:-4] + " " + root + '/' + f)
            file_mix.write('\n')

    file_aux = open(aux_scp, 'w')
    for root, dirs, files in os.walk(aux):
        files.sort()
        for f in files:
            file_aux.write(f[:-4] + " " + root + '/' + f)
            file_aux.write('\n')

    file_s1 = open(ref_scp, 'w')
    for root, dirs, files in os.walk(ref):
        files.sort()
        for f in files:
            file_s1.write(f[:-4] + " " + root + '/' + f)
            file_s1.write('\n')


def main(args):
    logging.basicConfig(level=logging.INFO)
    dataPath = args.data_dir
    genrateScp("train", dataPath)
    logging.info("Finished creating train scp")
    genrateScp("test", dataPath)
    logging.info("Finished creating test scp")
    genrateScp("dev", dataPath)
    logging.info("Finished creating cv scp")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to create mixtures scp in Kaldi wav.scp format'
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help='Input audio path'
    )

    # python 3_create_scp.py --data_dir "/Work21/2020/lijunjie/USSfromjingru/Junjie_Simulation/Libirmix/data/mixture/Librispeech_mixture"

    args = parser.parse_args()
    main(args)
