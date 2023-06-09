import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.models import save_publishable
from tools.tac_4dataset import TACDataset
# from tools.tac_dataset import TACDataset #2说话人
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

from Models.EaBNet_NS_ch16 import EaBNet
# from asteroid.models.fasnet import FasNetTAC
# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py.
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


class TACSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, valid_channels = batch
        # valid_channels contains a list of valid microphone channels for each example.
        # each example can have a varying number of microphone channels (can come from different arrays).
        # e.g. [[2], [4], [1]] three examples with 2 mics 4 mics and 1 mics.
        # print("inputs.shape:{}, targets.shape:{}".format(inputs.shape, targets.shape))
        # inputs = inputs[:, 2:, ...].contiguous()
        # targets = targets[:, 2:,...].contiguous()
        # print("After inputs.shape:{}, targets.shape:{}".format(inputs.shape, targets.shape))


        est_targets = self.model(inputs, valid_channels)
        # print("keys: ", )
        # pair_loss_func = pairwise_neg_sisdr
        # print("optimizer:{}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        # print("self.current_epoch:{}".format(self.current_epoch))
        # pair_loss = pair_loss_func(est_targets, targets[:, 0])
        loss = self.loss_func(est_targets, targets[:, 0]) #.mean()  # first channel is used as ref
        # print("pair_loss:{}".format(pair_loss))
        # print("pair_loss.shape:{}".format(pair_loss.shape))
        # print("pit loss:{}".format(loss))
        return loss


def main(conf):

    #不打乱顺序
    train_set = TACDataset(conf["data"]["train_json"], conf["data"]["segment"], train=False)
    val_set = TACDataset(conf["data"]["dev_json"], conf["data"]["segment"], train=False)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    #后续要修改一下参数列表
    # model = FasNetTAC(**conf["net"], sample_rate=conf["data"]["sample_rate"])
    model = EaBNet(n_src =conf["net"]["n_src"], sample_rate=conf["data"]["sample_rate"], M = 4)
    #model = TRUNet(12, 64)
    print("File location: {}".format(EaBNet))
    #print("File location: {}".format(TRUNet))
    # print("File: location:{}".format(FasNetTAC))
    print(model)
    print(model.get_model_args())
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    if conf["trick"]["half_lr"]:
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=conf["trick"]["lr_patience"]
        )
    else:
        scheduler = None
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr)
    system = TACSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=conf["training"]["save_top_k"],
        verbose=True,
    )
    callbacks.append(checkpoint)
    if conf["trick"]["early_stop"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", patience=conf["trick"]["es_patience"], verbose=True
            )
        )

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    if conf["training"]["resume"]:
        check_path = "/Work21/2021/wanghonglong/Nera/exp/train_TAC_eab_6mic/checkpoints/epoch=78-step=39499.ckpt"
        trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend="ddp",
        gradient_clip_val=conf["trick"]["gradient_clipping"],
        resume_from_checkpoint = check_path)
        ckpoint = torch.load(check_path)
        #print(checkpoint['state_dict'].keys())
        print("state_dict.keys:{}".format(ckpoint.keys()))
        print("lr_schedulers:{}".format(ckpoint['lr_schedulers']))
    else :
        trainer = pl.Trainer(
            max_epochs=conf["training"]["epochs"],
            callbacks=callbacks,
            default_root_dir=exp_dir,
            gpus=gpus,
            accelerator="ddp", #distributed_backend="ddp",
            gradient_clip_val=conf["trick"]["gradient_clipping"],
        )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    save_publishable(
        os.path.join(exp_dir, "publish_dir"),
        to_save,
        metrics=dict(),
        train_conf=conf,
        recipe="asteroid/TAC",
    )


if __name__ == "__main__":
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("./config/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    #打印时间
    import time
    nowt = time.ctime(time.time()) #服务器获取的系统时间好像就是格林尼治时间所以加上八小时
    print("Train: train.py.")
    print("Start Time:{}".format(nowt))
    print("Torch Version: {}".format(torch.__version__))
    if torch.cuda.device_count() > 0:
        print(str(torch.cuda.get_device_properties(0)))
    #-----------------
    main(arg_dic)
