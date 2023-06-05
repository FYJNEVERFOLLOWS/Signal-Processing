import os
import pandas

exp_workspace = "/Work21/2021/fuyanjie/pycode/libri_seg_spex+/data"



def filter():
    splits = ['dev', 'train']
    wav_scp_dir = os.path.join(exp_workspace, 'libri_seg_extr')

    for split in splits:
        metadata_path = f"/Work21/2021/fuyanjie/pycode/libri_seg_spex+/data/libri_seg_extr/{split}/Librispeech_{split}_2mixture_metadata.csv"
        metadata = pandas.read_csv(metadata_path)
        metadata.columns = metadata.columns.str.strip()
        print(f"{metadata}")

        mix_scp = os.path.join(wav_scp_dir, split, 'mix1.scp')
        aux_scp = os.path.join(wav_scp_dir, split, 'aux1.scp')
        ref_scp = os.path.join(wav_scp_dir, split, 'ref1.scp')

        file_mix = open(mix_scp, 'w')
        file_aux = open(aux_scp, 'w')
        file_s1 = open(ref_scp, 'w')

        lines = open(os.path.join(wav_scp_dir, split, 'ref.scp'), 'r').readlines()
        for line in lines:
            line = line.strip()
            uttid_line, path_line = line.split(' ', 1)
            # print(f'uttid_line {uttid_line} path_line {path_line}')
            audio_len = metadata[metadata['mixture_ID'] == uttid_line]['audio_length']
            if int(audio_len) > 64000:
                continue
            file_mix.write(uttid_line + ' ' + path_line + '\n')
            file_aux.write(uttid_line + ' ' + path_line + '\n')
            file_s1.write(uttid_line + ' ' + path_line + '\n')


if __name__ == '__main__':
    filter()
