import pandas
import numpy as np

metadata_path = "/Work21/2020/lijunjie/USSfromjingru/Junjie_Simulation/Libirmix/data/metadata/Librispeech_dev_2mixture_metadata.csv"
metadata = pandas.read_csv(metadata_path)

print(f"metadata['source_1_end'].astype(int) {metadata['source_1_end'].tolist()} \n metadata['source_2_end'].astype(int) {metadata['source_1_start'].tolist()}")

source_1_starts = metadata['source_1_start'].tolist()
source_1_ends = metadata['source_1_end'].tolist()
source_2_starts = metadata['source_2_start'].tolist()
source_2_ends = metadata['source_2_end'].tolist()
overlap_ratio_group_cnt = np.zeros(6)

for i in range(len(source_1_starts)):
    ss = min(source_1_ends[i], source_2_ends[i]) - max(source_1_starts[i], source_2_starts[i])
    non_silence = max(source_1_ends[i], source_2_ends[i]) - min(source_1_starts[i], source_2_starts[i])

    print(f'ss {ss} non_silence {non_silence} ss / non_silence {ss / non_silence}')

    if ss <= 0:
        overlap_ratio_group_cnt[0] += 1
    elif ss / non_silence > 0.0 and ss / non_silence <= 0.2:
        overlap_ratio_group_cnt[1] += 1
    elif ss / non_silence > 0.2 and ss / non_silence <= 0.4:
        overlap_ratio_group_cnt[2] += 1
    elif ss / non_silence > 0.4 and ss / non_silence <= 0.6:
        overlap_ratio_group_cnt[3] += 1
    elif ss / non_silence > 0.6 and ss / non_silence <= 0.8:
        overlap_ratio_group_cnt[4] += 1
    elif ss / non_silence > 0.8 and ss / non_silence <= 1.0:
        overlap_ratio_group_cnt[5] += 1

print(f'0%: {overlap_ratio_group_cnt[0]}\n'
      f'0-20%: {overlap_ratio_group_cnt[1]}\n'
      f'20-40%: {overlap_ratio_group_cnt[2]}\n'
      f'40-60%: {overlap_ratio_group_cnt[3]}\n'
      f'60-80%: {overlap_ratio_group_cnt[4]}\n'
      f'80-100%: {overlap_ratio_group_cnt[5]}\n')

print(f'TOTAL CLIPS: {np.sum(overlap_ratio_group_cnt)}')