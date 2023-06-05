fs = 8000
chunk_len = 4  # (s)
chunk_size = chunk_len * fs
num_spks = 1

# network configure
nnet_conf = {
    "L": 20,
    "N": 256,
    "X": 8,
    "R": 4,
    "B": 256,
    "H": 512,
    "P": 3,
    "norm": "gLN",
    "num_spks": num_spks,
    "non_linear": "relu"
}

# data configure:
train_dir = "/Work21/2021/fuyanjie/pycode/libri_seg_spex+/data/libri_seg_extr/train/"
dev_dir = "/Work21/2021/fuyanjie/pycode/libri_seg_spex+/data/libri_seg_extr/dev/"
spk_list = "/Work21/2021/fuyanjie/pycode/wsj0_seg_spex+_baseline/data/wsj0_2mix_extr_tr.spk"

train_data = {
    "mix_scp":
    train_dir + "mix.scp",
    "ref_scp":
    train_dir + "ref.scp",
    "aux_scp":
    train_dir + "aux.scp",
    "metadata_path":
    train_dir + "Librispeech_train_2mixture_metadata.csv",
    "spk_list": spk_list,
    "sample_rate": fs,
}

dev_data = {
    "mix_scp": 
    dev_dir + "mix.scp",
    "ref_scp":
    dev_dir + "ref.scp",
    "aux_scp":
    dev_dir + "aux.scp",
    "metadata_path":
    dev_dir + "Librispeech_dev_2mixture_metadata.csv",
    "spk_list": spk_list,
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "logging_period": 200  # batch number
}
