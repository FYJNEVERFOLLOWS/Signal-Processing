import os
import sys
import time

from itertools import permutations
from collections import defaultdict

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from .utils import get_logger
from libs.audio import write_wav

#from pudb import set_trace
#set_trace()


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.timer = SimpleTimer()

    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.2f})...".format(N, avg))

    def report(self, inference=False, details=False):
        N = len(self.loss)
        stat = dict()
        if inference:
            # non-decreasing order
            self.loss.sort()
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            for quantile in quantiles:
                part = self.loss[:N * quantile]
                part_mean = np.mean(part)
                part_std = np.std(part)
                res = dict()
                res["mean"] = part_mean
                res["std"] = part_std
                stat[quantile] = res
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "loss_std": np.std(self.loss),
            "batches": N,
            "cost": self.timer.elapsed(),
            "statistics": stat
        }


class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 no_impr=6,
                 sample_rate=8000):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        log_path = os.path.join(checkpoint, "trainer.log")
        if os.path.exists(log_path):
            os.remove(log_path)
        self.logger = get_logger(log_path, file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr
        self.sample_rate = sample_rate

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            ###
            #egs['aux_mfcc'] = list(egs['aux_mfcc'])
            ###

            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs, True)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

            reporter.add(loss.item())
            del egs, loss
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                ###
                #egs['aux_mfcc'] = list(egs['aux_mfcc'])
                ###
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs, False)
                reporter.add(loss.item())
                del egs, loss
        return reporter.report(details=True)

    def run(self, train_loader, dev_loader, num_epochs=50):
        # avoid alloc memory from gpu0
        with th.cuda.device(self.gpuid[0]):
            stats = dict()
            # # check if save is OK
            # self.save_checkpoint(best=False)
            # cv = self.eval(dev_loader)
            # best_loss = cv["loss"]
            # self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
            #     self.cur_epoch, best_loss))
            # no_impr = 0
            # # make sure not inf
            # self.scheduler.best = best_loss
            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                stats[
                    "title"] = "Loss(lr={:.3e}) - Epoch {:2d}:".format(
                        cur_lr, self.cur_epoch)
                tr = self.train(train_loader)
                stats["tr"] = "train_loss_mean = {:+.4f} train_loss_std = {:+.4f} ({:.2f} mins / {:d} batches)".format(
                    tr["loss"], tr["loss_std"], tr["cost"], tr["batches"])
                cv = self.eval(dev_loader)
                stats["cv"] = "dev_loss_mean = {:+.4f} dev_loss_std = {:+.4f} ({:.2f} mins / {:d} batches)".format(
                    cv["loss"], cv["loss_std"], cv["cost"], cv["batches"])
                stats["scheduler"] = ""
                if self.cur_epoch > 1 and cv["loss"] > best_loss:
                    no_impr += 1
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        self.scheduler.best)
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(best=True)
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                # schedule here
                self.scheduler.step(cv["loss"])
                # flush scheduler info
                sys.stdout.flush()
                # save last checkpoint
                self.save_checkpoint(best=False)
                if no_impr == self.no_impr:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, num_epochs))



class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def l2norm(self, mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    def sdr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, B x S tensor
        s: reference signal, B x S tensor
        Return:
        sdr: B tensor
        """
        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True) # zero-mean norm
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        noise = x_zm - s_zm
        return 10 * th.log10(th.pow(self.l2norm(s_zm), 2) / (th.pow(self.l2norm(noise), 2) + eps) + eps)

    def wsdr(self, x1, t1, y, eps=1e-8):
        """
        Arguments:
        x1: separated signal, B x T tensor
        t1: reference signal, B x T tensor
        y: mixture signal, B x T tensor
        Return:
        wsdr: B tensor
        """
        if x1.shape != t1.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x1.shape, t1.shape))
        if len(x1.shape) == 1:
            x1 = th.unsqueeze(x1, 0) # [1, T]
            t1 = th.unsqueeze(t1, 0)
            y = th.unsqueeze(y, 0)

        x2 = y - x1
        t2 = y - t1
        t1_norm = th.norm(t1, dim=-1)
        t2_norm = th.norm(t2, dim=-1)
        x1_norm = th.norm(x1, dim=-1)
        x2_norm = th.norm(x2, dim=-1)
        def dotproduct(y, y_hat) :
            #batch x channel x nsamples
            return th.bmm(y.reshape(y.shape[0], 1, y.shape[-1]), y_hat.reshape(y_hat.shape[0], y_hat.shape[-1], 1)).reshape(-1)
        def loss_sdr(a, a_hat,  a_norm, a_hat_norm):
            return dotproduct(a, a_hat) / (a_norm * a_hat_norm + eps)

        alpha = t1_norm.pow(2) / (t1_norm.pow(2) + t2_norm.pow(2) + eps)
        loss_wSDR = -alpha * loss_sdr(t1, x1, t1_norm, x1_norm) - (1 - alpha) * loss_sdr(t2, x2, t2_norm, x2_norm)

        return loss_wSDR

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """
        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True) # zero-mean norm
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (self.l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + self.l2norm(t) / (self.l2norm(x_zm - t) + eps))

    def loss_energy(self, x, eps=1e-8):
        """
        Arguments:
        x: estimated signal, S tensor
        average by duration (seconds)
        """
        return 10 * th.log10(th.pow(self.l2norm(x), 2) * self.sample_rate / x.shape[0] + eps)

    def mask_by_length(self, xs, lengths, fill=0):
        """Mask tensor according to length.

        Args:
            xs (Tensor): Batch of input tensor (B, `*`).
            lengths (LongTensor or List): Batch of lengths (B,).
            fill (int or float): Value to fill masked part.

        Returns:
            Tensor: Batch of masked input tensor (B, `*`).

        Examples:
            >>> x = torch.arange(5).repeat(3, 1) + 1
            >>> x
            tensor([[1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5]])
            >>> lengths = [5, 3, 2]
            >>> mask_by_length(x, lengths)
            tensor([[1, 2, 3, 4, 5],
                    [1, 2, 3, 0, 0],
                    [1, 2, 0, 0, 0]])

        """
        assert xs.size(0) == len(lengths)
        ret = xs.data.new(*xs.size()).fill_(fill)
        for i, l in enumerate(lengths):
            ret[i, :l] = xs[i, :l]
        return ret

    # # utt-level SI-SDR only (without ce loss & energy loss)
    # def compute_loss(self, egs, train_or_valid):
    #     ests, ests2, ests3, spk_pred = th.nn.parallel.data_parallel(
    #         self.nnet, (egs["mix"], egs["aux"], egs["aux_len"]), device_ids=self.gpuid)
    #     refs = egs["ref"]

    #     ## P x B
    #     B = egs["mix"].size(0)
    #     # valid_len = [egs['audio_max_len']] * B
    #     audio_lens = [egs['audio_max_len'], ests.shape[1], ests2.shape[1], ests3.shape[1]]
    #     audio_max_len = np.max(audio_lens)

    #     padding_data = F.pad(ests, (0, audio_max_len-ests.shape[1]), 'constant', 0)
    #     ests = padding_data
    #     padding_data = F.pad(ests2, (0, audio_max_len-ests2.shape[1]), 'constant', 0)
    #     ests2 = padding_data
    #     padding_data = F.pad(ests3, (0, audio_max_len-ests3.shape[1]), 'constant', 0)
    #     ests3 = padding_data

    #     snr1 = self.sisnr(ests, refs)
    #     snr2 = self.sisnr(ests2, refs)
    #     snr3 = self.sisnr(ests3, refs)
    #     snr_loss = (-0.8*th.sum(snr1)-0.1*th.sum(snr2)-0.1*th.sum(snr3)) / B
 
    #     return snr_loss

    # # utt-level SDR only (without ce loss & energy loss)
    # def compute_loss(self, egs, train_or_valid):
    #     ests, ests2, ests3, spk_pred = th.nn.parallel.data_parallel(
    #         self.nnet, (egs["mix"], egs["aux"], egs["aux_len"]), device_ids=self.gpuid)
    #     refs = egs["ref"]

    #     ## P x B
    #     B = egs["mix"].size(0)
    #     # valid_len = [egs['audio_max_len']] * B
    #     audio_lens = [egs['audio_max_len'], ests.shape[1], ests2.shape[1], ests3.shape[1]]
    #     audio_max_len = np.max(audio_lens)

    #     padding_data = F.pad(ests, (0, audio_max_len-ests.shape[1]), 'constant', 0)
    #     ests = padding_data
    #     padding_data = F.pad(ests2, (0, audio_max_len-ests2.shape[1]), 'constant', 0)
    #     ests2 = padding_data
    #     padding_data = F.pad(ests3, (0, audio_max_len-ests3.shape[1]), 'constant', 0)
    #     ests3 = padding_data

    #     snr1 = self.sdr(ests, refs)
    #     snr2 = self.sdr(ests2, refs)
    #     snr3 = self.sdr(ests3, refs)
    #     snr_loss = (-0.8*th.sum(snr1)-0.1*th.sum(snr2)-0.1*th.sum(snr3)) / B
 
    #     return snr_loss

    # # utt-level wSDR only (without ce loss & energy loss)
    # def compute_loss(self, egs, train_or_valid):
    #     ests, ests2, ests3, spk_pred = th.nn.parallel.data_parallel(
    #         self.nnet, (egs["mix"], egs["aux"], egs["aux_len"]), device_ids=self.gpuid)
    #     refs = egs["ref"]
    #     mixs = egs["mix"]

    #     ## P x B
    #     B = egs["mix"].size(0)
    #     audio_lens = [egs['audio_max_len'], ests.shape[1], ests2.shape[1], ests3.shape[1]]
    #     audio_max_len = np.max(audio_lens)

    #     padding_data = F.pad(ests, (0, audio_max_len-ests.shape[1]), 'constant', 0)
    #     ests = padding_data
    #     padding_data = F.pad(ests2, (0, audio_max_len-ests2.shape[1]), 'constant', 0)
    #     ests2 = padding_data
    #     padding_data = F.pad(ests3, (0, audio_max_len-ests3.shape[1]), 'constant', 0)
    #     ests3 = padding_data

    #     wsdr1 = self.wsdr(ests, refs, mixs)
    #     wsdr2 = self.wsdr(ests2, refs, mixs)
    #     wsdr3 = self.wsdr(ests3, refs, mixs)
    #     wsdr_loss = (0.8*th.sum(wsdr1)+0.1*th.sum(wsdr2)+0.1*th.sum(wsdr3)) / B
 
    #     return wsdr_loss

    # # seg-level wSDR and Energy
    # def compute_loss(self, egs, train_or_valid):
    #     ests, ests2, ests3, spk_pred = th.nn.parallel.data_parallel(
    #         self.nnet, (egs["mix"], egs["aux"], egs["aux_len"]), device_ids=self.gpuid)
    #     refs = egs["ref"]
    #     mixs = egs["mix"]
    #     audio_max_len = egs["audio_max_len"]
    #     # mix / ref: [B, T]
    #     # aux: [B, max_aux_len] aux_len: [B]
    #     # ests / ests2 / ests3: [B, T]
    #     padding_data = F.pad(ests, (0, audio_max_len-ests.shape[1]), 'constant', 0)
    #     ests = padding_data
    #     padding_data = F.pad(ests2, (0, audio_max_len-ests2.shape[1]), 'constant', 0)
    #     ests2 = padding_data
    #     padding_data = F.pad(ests3, (0, audio_max_len-ests3.shape[1]), 'constant', 0)
    #     ests3 = padding_data

    #     B = egs["mix"].size(0)
    #     loss_sum = None

    #     if not train_or_valid:
    #         wsdr1 = self.wsdr(ests, refs, mixs)
    #         wsdr2 = self.wsdr(ests2, refs, mixs)
    #         wsdr3 = self.wsdr(ests3, refs, mixs)
    #         loss_sum = 0.8*th.sum(wsdr1)+0.1*th.sum(wsdr2)+0.1*th.sum(wsdr3)
    #     else:
    #         for batch_idx in range(B):
    #             source_1_start = egs["source_1_start"][batch_idx]
    #             source_1_end = egs["source_1_end"][batch_idx]
    #             source_2_start = egs["source_2_start"][batch_idx]
    #             source_2_end = egs["source_2_end"][batch_idx] 
    #             # print(f'audio_max_len {audio_max_len} source_1_start {source_1_start} source_1_end {source_1_end}', flush=True)

    #             mix_s1_present = mixs[batch_idx, source_1_start:source_1_end]
    #             mix_s1_absent_1 = mixs[batch_idx, :source_1_start]
    #             mix_s1_absent_2 = mixs[batch_idx, source_1_end:]
    #             ref_s1_present = refs[batch_idx, source_1_start:source_1_end]

    #             ests_s1_present = ests[batch_idx, source_1_start:source_1_end]
    #             ests_s1_absent_1 = ests[batch_idx, :source_1_start]
    #             ests_s1_absent_2 = ests[batch_idx, source_1_end:]
    #             ests2_s1_present = ests2[batch_idx, source_1_start:source_1_end]
    #             ests2_s1_absent_1 = ests2[batch_idx, :source_1_start]
    #             ests2_s1_absent_2 = ests2[batch_idx, source_1_end:]
    #             ests3_s1_present = ests3[batch_idx, source_1_start:source_1_end]
    #             ests3_s1_absent_1 = ests3[batch_idx, :source_1_start]
    #             ests3_s1_absent_2 = ests3[batch_idx, source_1_end:]

    #             wsdr1 = self.wsdr(ests_s1_present, ref_s1_present, mix_s1_present)
    #             wsdr2 = self.wsdr(ests2_s1_present, ref_s1_present, mix_s1_present)
    #             wsdr3 = self.wsdr(ests3_s1_present, ref_s1_present, mix_s1_present)
    #             wsdr_sum = 0.8*wsdr1 + 0.1*wsdr2 + 0.1*wsdr3 
                
    #             ests_loss_energy_1 = 0 if ests_s1_absent_1.nelement() == 0 else self.loss_energy(ests_s1_absent_1)
    #             ests_loss_energy_2 = 0 if ests_s1_absent_2.nelement() == 0 else self.loss_energy(ests_s1_absent_2)
    #             ests2_loss_energy_1 = 0 if ests2_s1_absent_1.nelement() == 0 else self.loss_energy(ests2_s1_absent_1)
    #             ests2_loss_energy_2 = 0 if ests2_s1_absent_2.nelement() == 0 else self.loss_energy(ests2_s1_absent_2)
    #             ests3_loss_energy_1 = 0 if ests3_s1_absent_1.nelement() == 0 else self.loss_energy(ests3_s1_absent_1)
    #             ests3_loss_energy_2 = 0 if ests3_s1_absent_2.nelement() == 0 else self.loss_energy(ests3_s1_absent_2)

    #             loss_energy = 0.8*(ests_loss_energy_1+ests_loss_energy_2) + 0.1*(ests2_loss_energy_1+ests2_loss_energy_2) + 0.1*(ests3_loss_energy_1+ests3_loss_energy_2)
    #             if batch_idx == 0:
    #                 print(f'loss_energy {loss_energy} sdr_sum {wsdr_sum}', flush=True)
    #                 loss_sum = 0.0005 * loss_energy + wsdr_sum
    #             else:
    #                 loss_sum += 0.0005 * loss_energy + wsdr_sum

    #         del wsdr1, wsdr2, wsdr3, wsdr_sum, ests, ests2, ests3, refs

    #     return loss_sum / B

    # seg-level SDR and Energy
    def compute_loss(self, egs, train_or_valid):
        ests, ests2, ests3, spk_pred = th.nn.parallel.data_parallel(
            self.nnet, (egs["mix"], egs["aux"], egs["aux_len"]), device_ids=self.gpuid)
        refs = egs["ref"]
        mixs = egs["mix"]
        audio_max_len = egs["audio_max_len"]
        # mix / ref: [B, T]
        # aux: [B, max_aux_len] aux_len: [B]
        # ests / ests2 / ests3: [B, T]
        padding_data = F.pad(ests, (0, audio_max_len-ests.shape[1]), 'constant', 0)
        ests = padding_data
        padding_data = F.pad(ests2, (0, audio_max_len-ests2.shape[1]), 'constant', 0)
        ests2 = padding_data
        padding_data = F.pad(ests3, (0, audio_max_len-ests3.shape[1]), 'constant', 0)
        ests3 = padding_data

        B = egs["mix"].size(0)
        loss_sum = None

        if not train_or_valid:
            snr1 = self.sdr(ests, refs)
            snr2 = self.sdr(ests2, refs)
            snr3 = self.sdr(ests3, refs)
            loss_sum = -0.8*th.sum(snr1)-0.1*th.sum(snr2)-0.1*th.sum(snr3)
        else:
            for batch_idx in range(B):
                source_1_start = egs["source_1_start"][batch_idx]
                source_1_end = egs["source_1_end"][batch_idx]
                source_2_start = egs["source_2_start"][batch_idx]
                source_2_end = egs["source_2_end"][batch_idx] 
                # print(f'audio_max_len {audio_max_len} source_1_start {source_1_start} source_1_end {source_1_end}', flush=True)

                mix_s1_present = mixs[batch_idx, source_1_start:source_1_end]
                mix_s1_absent_1 = mixs[batch_idx, :source_1_start]
                mix_s1_absent_2 = mixs[batch_idx, source_1_end:]
                ref_s1_present = refs[batch_idx, source_1_start:source_1_end]

                ests_s1_present = ests[batch_idx, source_1_start:source_1_end]
                ests_s1_absent_1 = ests[batch_idx, :source_1_start]
                ests_s1_absent_2 = ests[batch_idx, source_1_end:]
                ests2_s1_present = ests2[batch_idx, source_1_start:source_1_end]
                ests2_s1_absent_1 = ests2[batch_idx, :source_1_start]
                ests2_s1_absent_2 = ests2[batch_idx, source_1_end:]
                ests3_s1_present = ests3[batch_idx, source_1_start:source_1_end]
                ests3_s1_absent_1 = ests3[batch_idx, :source_1_start]
                ests3_s1_absent_2 = ests3[batch_idx, source_1_end:]

                snr1 = self.sdr(ests_s1_present, ref_s1_present)
                snr2 = self.sdr(ests2_s1_present, ref_s1_present)
                snr3 = self.sdr(ests3_s1_present, ref_s1_present)
                sdr_sum = -0.8*snr1 - 0.1*snr2 - 0.1*snr3 
                
                ests_loss_energy_1 = 0 if ests_s1_absent_1.nelement() == 0 else self.loss_energy(ests_s1_absent_1)
                ests_loss_energy_2 = 0 if ests_s1_absent_2.nelement() == 0 else self.loss_energy(ests_s1_absent_2)
                ests2_loss_energy_1 = 0 if ests2_s1_absent_1.nelement() == 0 else self.loss_energy(ests2_s1_absent_1)
                ests2_loss_energy_2 = 0 if ests2_s1_absent_2.nelement() == 0 else self.loss_energy(ests2_s1_absent_2)
                ests3_loss_energy_1 = 0 if ests3_s1_absent_1.nelement() == 0 else self.loss_energy(ests3_s1_absent_1)
                ests3_loss_energy_2 = 0 if ests3_s1_absent_2.nelement() == 0 else self.loss_energy(ests3_s1_absent_2)

                loss_energy = 0.8*(ests_loss_energy_1+ests_loss_energy_2) + 0.1*(ests2_loss_energy_1+ests2_loss_energy_2) + 0.1*(ests3_loss_energy_1+ests3_loss_energy_2)
                if batch_idx == 0:
                    loss_sum = 0.005 * loss_energy + sdr_sum
                    print(f'loss_energy {loss_energy} sdr_sum {sdr_sum}', flush=True)
                else:
                    loss_sum += 0.005 * loss_energy + sdr_sum

            del snr1, snr2, snr3, sdr_sum, ests, ests2, ests3, refs

        return loss_sum / B

    # # frame-level SDR and Energy
    # def compute_loss(self, egs, train_or_valid):
    #     ests, ests2, ests3, spk_pred = th.nn.parallel.data_parallel(
    #         self.nnet, (egs["mix"], egs["aux"], egs["aux_len"]), device_ids=self.gpuid)
    #     refs = egs["ref"]
    #     mixs = egs["mix"]
    #     audio_max_len = egs["audio_max_len"]
    #     # mix / ref: [B, T]
    #     # aux: [B, max_aux_len] aux_len: [B]
    #     # ests / ests2 / ests3: [B, T]
    #     padding_data = F.pad(ests, (0, audio_max_len-ests.shape[1]), 'constant', 0)
    #     ests = padding_data
    #     padding_data = F.pad(ests2, (0, audio_max_len-ests2.shape[1]), 'constant', 0)
    #     ests2 = padding_data
    #     padding_data = F.pad(ests3, (0, audio_max_len-ests3.shape[1]), 'constant', 0)
    #     ests3 = padding_data

    #     B = egs["mix"].size(0)
    #     loss_sum = None

    #     if not train_or_valid:
    #         snr1 = self.sdr(ests, refs)
    #         snr2 = self.sdr(ests2, refs)
    #         snr3 = self.sdr(ests3, refs)
    #         loss_sum = -0.8*th.sum(snr1)-0.1*th.sum(snr2)-0.1*th.sum(snr3)
    #     else:
    #         for batch_idx in range(B):
    #             source_1_start = egs["source_1_start"][batch_idx]
    #             source_1_end = egs["source_1_end"][batch_idx]
    #             source_2_start = egs["source_2_start"][batch_idx]
    #             source_2_end = egs["source_2_end"][batch_idx] 
    #             # print(f'audio_max_len {audio_max_len} source_1_start {source_1_start} source_1_end {source_1_end}', flush=True)

    #             mix_s1_present = mixs[batch_idx, source_1_start:source_1_end]
    #             mix_s1_absent_1 = mixs[batch_idx, :source_1_start]
    #             mix_s1_absent_2 = mixs[batch_idx, source_1_end:]
    #             ref_s1_present = refs[batch_idx, source_1_start:source_1_end]

    #             ests_s1_present = ests[batch_idx, source_1_start:source_1_end]
    #             ests_s1_absent_1 = ests[batch_idx, :source_1_start]
    #             ests_s1_absent_2 = ests[batch_idx, source_1_end:]
    #             ests2_s1_present = ests2[batch_idx, source_1_start:source_1_end]
    #             ests2_s1_absent_1 = ests2[batch_idx, :source_1_start]
    #             ests2_s1_absent_2 = ests2[batch_idx, source_1_end:]
    #             ests3_s1_present = ests3[batch_idx, source_1_start:source_1_end]
    #             ests3_s1_absent_1 = ests3[batch_idx, :source_1_start]
    #             ests3_s1_absent_2 = ests3[batch_idx, source_1_end:]

    #             N = len(ref_s1_present) // 512
    #             sdr_sum = None
    #             for frame_idx in range(N):
    #                 snr1 = self.sdr(ests_s1_present[frame_idx * 512:(frame_idx+1) * 512], ref_s1_present[frame_idx * 512:(frame_idx+1) * 512])
    #                 snr2 = self.sdr(ests2_s1_present[frame_idx * 512:(frame_idx+1) * 512], ref_s1_present[frame_idx * 512:(frame_idx+1) * 512])
    #                 snr3 = self.sdr(ests3_s1_present[frame_idx * 512:(frame_idx+1) * 512], ref_s1_present[frame_idx * 512:(frame_idx+1) * 512])
    #                 if frame_idx == 0:
    #                     sdr_sum = -0.8*snr1 - 0.1*snr2 - 0.1*snr3 
    #                 else:
    #                     sdr_sum += -0.8*snr1 - 0.1*snr2 - 0.1*snr3 
    #             sdr_sum /= N
    #             ests_loss_energy_1 = 0 if ests_s1_absent_1.nelement() == 0 else self.loss_energy(ests_s1_absent_1)
    #             ests_loss_energy_2 = 0 if ests_s1_absent_2.nelement() == 0 else self.loss_energy(ests_s1_absent_2)
    #             ests2_loss_energy_1 = 0 if ests2_s1_absent_1.nelement() == 0 else self.loss_energy(ests2_s1_absent_1)
    #             ests2_loss_energy_2 = 0 if ests2_s1_absent_2.nelement() == 0 else self.loss_energy(ests2_s1_absent_2)
    #             ests3_loss_energy_1 = 0 if ests3_s1_absent_1.nelement() == 0 else self.loss_energy(ests3_s1_absent_1)
    #             ests3_loss_energy_2 = 0 if ests3_s1_absent_2.nelement() == 0 else self.loss_energy(ests3_s1_absent_2)

    #             loss_energy = 0.8*(ests_loss_energy_1+ests_loss_energy_2) + 0.1*(ests2_loss_energy_1+ests2_loss_energy_2) + 0.1*(ests3_loss_energy_1+ests3_loss_energy_2)
    #             if batch_idx == 0:
    #                 loss_sum = 0.005 * loss_energy + sdr_sum
    #                 print(f'loss_energy {loss_energy} sdr_sum {sdr_sum}', flush=True)
    #             else:
    #                 loss_sum += 0.005 * loss_energy + sdr_sum

    #         del snr1, snr2, snr3, sdr_sum, ests, ests2, ests3, refs

    #     return loss_sum / B

    # # Reweight SI-SNR and Energy
    # def compute_loss(self, egs):
    #     ests, ests2, ests3, spk_pred = th.nn.parallel.data_parallel(
    #         self.nnet, (egs["mix"], egs["aux"], egs["aux_len"]), device_ids=self.gpuid)
    #     refs = egs["ref"]
    #     mixs = egs["mix"]

    #     # mix / ref: [B, T]
    #     # aux: [B, max_aux_len] aux_len: [B]
    #     # ests / ests2 / ests3: [B, T]
    #     B = egs["mix"].size(0)
    #     valid_len = egs["valid_len"]
    #     print(f'egs["mix"] {egs["mix"].shape}')
    #     print(f'egs["ref"] {refs.shape}')
    #     print(f'egs["aux"] {egs["aux"].shape}')
    #     print(f'egs["aux_len"] {egs["aux_len"].shape}')
    #     print(f'ests {ests.shape}')
    #     print(f'ests2 {ests2.shape}')
    #     print(f'ests3 {ests3.shape}')

    #     for batch_idx in range(B):
    #         source_1_start = egs["source_1_start"][batch_idx]
    #         source_1_end = egs["source_1_end"][batch_idx]
    #         source_2_start = egs["source_2_start"][batch_idx]
    #         source_2_end = egs["source_2_end"][batch_idx] 
    #         print(f'key {egs["key"][batch_idx]} source_1_start: {source_1_start} source_1_end: {source_1_end}')
    #         print(f'key {egs["key"][batch_idx]} source_2_start: {source_2_start} source_2_end: {source_2_end}')

    #         mix_s1_present = mixs[batch_idx, source_1_start:source_1_end, :]
    #         mix_s1_absent_1 = mixs[batch_idx, :source_1_start, :]
    #         mix_s1_absent_2 = mixs[batch_idx, source_1_end:, :]
    #         ref_s1_present = refs[batch_idx, :(source_1_end-source_1_start), :]

    #         print(f'key {egs["key"][batch_idx]} mix_s1_present: {mix_s1_present.shape}')
    #         print(f'key {egs["key"][batch_idx]} mix_s1_absent_1: {mix_s1_absent_1.shape} mix_s1_absent_2: {mix_s1_absent_2.shape}')
    #         ests_s1_present = ests[batch_idx, source_1_start:source_1_end, :]
    #         ests_s1_absent_1 = ests[batch_idx, :source_1_start, :]
    #         ests_s1_absent_2 = ests[batch_idx, source_1_end:, :]
    #         ests2_s1_present = ests2[batch_idx, source_1_start:source_1_end, :]
    #         ests2_s1_absent_1 = ests2[batch_idx, :source_1_start, :]
    #         ests2_s1_absent_2 = ests2[batch_idx, source_1_end:, :]
    #         ests3_s1_present = ests3[batch_idx, source_1_start:source_1_end, :]
    #         ests3_s1_absent_1 = ests3[batch_idx, :source_1_start, :]
    #         ests3_s1_absent_2 = ests3[batch_idx, source_1_end:, :]
    #         snr1 = self.sisnr(ests_s1_present, ref_s1_present)
    #         snr2 = self.sisnr(ests2_s1_present, ref_s1_present)
    #         snr3 = self.sisnr(ests3_s1_present, ref_s1_present)
    #         snr_sum = -0.8*snr1 - 0.1*snr2 - 0.1*snr3 # snr_sum.shape: [B]
            
    #         ests_loss_energy_1 = self.loss_energy(ests_s1_absent_1)
    #         ests_loss_energy_2 = self.loss_energy(ests_s1_absent_2)
    #         ests2_loss_energy_1 = self.loss_energy(ests2_s1_absent_1)
    #         ests2_loss_energy_2 = self.loss_energy(ests2_s1_absent_2)
    #         ests3_loss_energy_1 = self.loss_energy(ests3_s1_absent_1)
    #         ests3_loss_energy_2 = self.loss_energy(ests3_s1_absent_2)

    #         loss_energy = 0.8*(ests_loss_energy_1+ests_loss_energy_2) + 0.1*(ests2_loss_energy_1+ests2_loss_energy_2) + 0.1*(ests3_loss_energy_1+ests3_loss_energy_2)
    #         print(f'snr_sum {snr_sum} loss_energy: {loss_energy}')


    #     F = th.ones(snr_sum.shape) # [B]
    #     # print("snr_sum: ", snr_sum, flush=True)

    #     alpha = 0.067
    #     F = snr_sum * alpha
    #     F = F.flatten(0)
    #     # print("F: ", F, flush=True)

    #     new_weights = th.softmax(F, 0).cuda() # new_weights.shape: [B]
    #     snr_sum = new_weights * snr_sum.flatten(0)
    #     # print("new_weights: {}\n-----\n".format(new_weights), flush=True)
    #     # snr_loss = th.sum(snr_sum) / -B # mean for unbiased mini-batch 
    #     snr_loss = th.sum(snr_sum) # sum for reweighting biased training

    #     # del snr1, snr2, snr3, snr_sum, F, new_weights, ests, ests2, ests3, refs

    #     return snr_loss

