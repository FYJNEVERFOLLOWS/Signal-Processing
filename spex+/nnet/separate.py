#!/usr/bin/env python

# wujian@2018

import os
import argparse

import torch as th
import numpy as np

from conv_tas_net_decode import ConvTasNet

from libs.utils import load_json, get_logger
from libs.metric import si_snri
from libs.trainer import SimpleTimer
from libs.audio import WaveReader, write_wav

logger = get_logger(__name__)

#from pudb import set_trace
#set_trace()
class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.sisdri = []
        self.sisdr = []
        self.uttid2sisdr = dict()
        self.timer = SimpleTimer()

    def add(self, sisdri, sisdr, uttid):
        self.sisdri.append(sisdri)
        self.sisdr.append(sisdr)
        self.uttid2sisdr[uttid] = sisdr
        N = len(self.sisdri)
        if not N % self.period:
            avg_sisdri = sum(self.sisdri[-self.period:]) / self.period
            avg_sisdr = sum(self.sisdr[-self.period:]) / self.period
            std_sisdri = np.std(self.sisdri[-self.period:])
            std_sisdr = np.std(self.sisdr[-self.period:])

            self.logger.info("Processed {:d} samples, last {:d} samples' statistics:"
                             " (mean SI-SDRi = {:+.2f}, mean SI-SDR = {:+.2f})...".format(N, self.period, avg_sisdri, avg_sisdr))
            self.logger.info("Processed {:d} samples, last {:d} samples' statistics:"
                             " (std SI-SDRi = {:+.4f}, std SI-SDR = {:+.4f})...".format(N, self.period, std_sisdri, std_sisdr))



    def report(self, quantiles, summary=True, details=False):
        N = len(self.sisdri)
        stat = dict()
        if summary:
            # non-decreasing order
            self.sisdri.sort()
            self.sisdr.sort()
            sisdri_quantiles = np.percentile(self.sisdri, quantiles)
            sisdr_quantiles = np.percentile(self.sisdr, quantiles)
            stat["sisdri_quantiles"] = sisdri_quantiles
            stat["sisdr_quantiles"] = sisdr_quantiles

            total_sisdri_mean = np.mean(self.sisdri)
            total_sisdri_std = np.std(self.sisdri)
            stat["total_sisdri_mean"] = total_sisdri_mean
            stat["total_sisdri_std"] = total_sisdri_std

            total_sisdr_mean = np.mean(self.sisdr)
            total_sisdr_std = np.std(self.sisdr)
            stat["total_sisdr_mean"] = total_sisdr_mean
            stat["total_sisdr_std"] = total_sisdr_std
            self.logger.info("UTT 2 SI-SDR on {:d} sample: {}".format(N, self.uttid2sisdr))

            sistr = ", ".join(map(lambda f: "{:.2f}".format(f), self.sisdri))
            sstr = ", ".join(map(lambda f: "{:.2f}".format(f), self.sisdr))
            self.logger.info("SI-SDRi on {:d} samples: {}".format(N, sistr))
            self.logger.info("SI-SDR on {:d} sample: {}".format(N, sstr))
        # if details:
        #     self.logger.info("Processed {:d} samples".format(N))
        #     sistr = ", ".join(map(lambda f: "{:.2f}".format(f), self.sisdri))
        #     sstr = ", ".join(map(lambda f: "{:.2f}".format(f), self.sisdr))
        #     self.logger.info("SI-SDRi on {:d} samples: {}".format(N, sistr))
        #     self.logger.info("SI-SDR on {:d} sample: {}".format(N, sstr))

        return {
            "samples": N,
            "cost": self.timer.elapsed(),
            "statistics": stat
        }

class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid):
        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")
        nnet = self._load_nnet(cpt_dir)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def _load_nnet(self, cpt_dir):
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = ConvTasNet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        return nnet

    def compute(self, mix_samps, ref_samps, aux_samps, aux_samps_len, reporter, uttid):
        with th.no_grad():
            raw = th.tensor(mix_samps, dtype=th.float32, device=self.device)
            ref = th.tensor(ref_samps, dtype=th.float32, device=self.device)
            aux = th.tensor(aux_samps, dtype=th.float32, device=self.device)
            aux_len = th.tensor(aux_samps_len, dtype=th.float32, device=self.device)
            aux = aux.unsqueeze(0)
            sps, sps2, sps3, spk_pred = self.nnet(raw, aux, aux_len)

            sp_samps = np.squeeze(sps.detach().cpu().numpy())

            ### computer SI-SDRi and SI-SDR
            sisdri, sisdr = si_snri(sp_samps, ref.cpu().numpy(), raw.cpu().numpy())
            reporter.add(sisdri, sisdr, uttid)

            return [sp_samps], sisdr


def run(args):
    reporter = ProgressReporter(logger, period=100)
    quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    mix_input = WaveReader("data/libri_seg_extr/test_wer/mix.scp", sample_rate=8000)
    aux_input = WaveReader("data/libri_seg_extr/test_wer/aux.scp", sample_rate=8000)
    ref_input = WaveReader("data/libri_seg_extr/test_wer/ref.scp", sample_rate=8000)
    computer = NnetComputer(args.cpt_dir, args.gpuid)

    ###
    os.makedirs(os.path.join(args.separated_dir, "lt-20"), exist_ok=True)
    os.makedirs(os.path.join(args.separated_dir, "-20--15"), exist_ok=True)
    os.makedirs(os.path.join(args.separated_dir, "-15--10"), exist_ok=True)
    os.makedirs(os.path.join(args.separated_dir, "-10--5"), exist_ok=True)
    os.makedirs(os.path.join(args.separated_dir, "-5-0"), exist_ok=True)
    os.makedirs(os.path.join(args.separated_dir, "0-5"), exist_ok=True)
    os.makedirs(os.path.join(args.separated_dir, "5-10"), exist_ok=True)
    os.makedirs(os.path.join(args.separated_dir, "10-15"), exist_ok=True)
    os.makedirs(os.path.join(args.separated_dir, "15-20"), exist_ok=True)
    os.makedirs(os.path.join(args.separated_dir, "20+"), exist_ok=True)
    sisdr_subfolders = ["lt-20", "-20--15", "-15--10", "-10--5", "-5-0", "0-5", "5-10", "10-15", "15-20", "20+"]
    
    ###
    for idx, (key, mix_samps) in enumerate(mix_input):
        aux_samps = aux_input[key]
        ref_samps = ref_input[key]
        # logger.info("Compute on utterance {}...".format(key))

        spks, sisdr = computer.compute(mix_samps, ref_samps, aux_samps, len(aux_samps), reporter, key)
        if idx % 100 == 0:
            reporter.report(quantiles, summary=False, details=True)

        if sisdr < -20:
            group = 0
        elif sisdr >= -20 and sisdr < -15:
            group = 1
        elif sisdr >= -15 and sisdr < -10:
            group = 2
        elif sisdr >= -10 and sisdr < -5:
            group = 3
        elif sisdr >= -5 and sisdr < 0:
            group = 4
        elif sisdr >= 0 and sisdr < 5:
            group = 5
        elif sisdr >= 5 and sisdr < 10:
            group = 6
        elif sisdr >= 10 and sisdr < 15:
            group = 7
        elif sisdr >= 15 and sisdr < 20:
            group = 8
        else:
            group = 9

        norm = np.linalg.norm(mix_samps, np.inf)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))
            write_wav(
                os.path.join(args.separated_dir, sisdr_subfolders[group], "{}.wav".format(key)), 
                samps,
                fs=8000)
    logger.info("Compute over {:d} utterances".format(len(mix_input)))
    sum_report = reporter.report(quantiles, summary=True, details=True)
    logger.info("Totally inferenced {:d} samples, "
                "Time cost: {:.2f} mins".format(sum_report["samples"], sum_report["cost"]))
    logger.info("Total SI-SDRi mean {:.2f}, "
                "Total SI-SDRi std {:.4f}".format(sum_report["statistics"]["total_sisdri_mean"], sum_report["statistics"]["total_sisdri_std"]))
    logger.info("Total SI-SDR mean {:.2f}, "
                "Total SI-SDR std {:.4f}".format(sum_report["statistics"]["total_sisdr_mean"], sum_report["statistics"]["total_sisdr_std"]))
    logger.info("=========================== Dividison ===================================")
    logger.info("Statistics about SI-SDRi quantiles {}:".format(sum_report["statistics"]["sisdri_quantiles"]))
    logger.info("Statistics about SI-SDR quantiles {}:".format(sum_report["statistics"]["sisdr_quantiles"]))


if __name__ == "__main__":
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='Inference Experiment Argument Parser')
    # ===============================================
    parser.add_argument("--gpuid", type=int, help="""GPU ID""", default=0)
    parser.add_argument('--cpt_dir', type=str, help='path to checkpoint dir')
    parser.add_argument('--separated_dir', type=str, help='path to separated audio dir')

    args = parser.parse_args()                        
    run(args)
