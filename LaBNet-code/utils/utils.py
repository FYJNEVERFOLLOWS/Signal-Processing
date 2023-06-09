import os
import numpy as np
import torch
from pesq import pesq
import soundfile as sf
import librosa
from itertools import permutations, product


time_bins = 249
batch = 4
EPS = 1e-8

def write_wav(fname, samps, fs=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    #if normalize:
    #    samps = samps * MAX_INT16
    ## scipy.io.wavfile.write could write single/multi-channel files
    ## for multi-channel, accept ndarray [Nsamples, Nchannels]
    #if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
    #    samps = np.transpose(samps)
    #    samps = np.squeeze(samps)
    ## same as MATLAB and kaldi
    #samps_int16 = samps.astype(np.int16)
    #fdir = os.path.dirname(fname)
    #if fdir and not os.path.exists(fdir):
    #    os.makedirs(fdir)
    ## NOTE: librosa 0.6.0 seems could not write non-float narray
    ##       so use scipy.io.wavfile instead
    #wf.write(fname, fs, samps_int16)

    # wham and whamr mixture and clean data are float 32, can not use scipy.io.wavfile to read and write int16
    # change to soundfile to read and write, although reference speech is int16, soundfile still can read and outputs as float
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    sf.write(fname, samps, fs, subtype='FLOAT')
    
def doa_err_2_source(gt_azi_arr, es_as_1, es_as_2):
    """
    gt_azi_arr: # [B, T, S, n_avb_mics=2]
    es_as_1 / es_as_2: [B, T, n_avb_mics=2, F] F: feature dimension of the likelihood-based coding
    """

    gt_azi_1 = gt_azi_arr[:, :, 0, :] # B, T, n_avb_mics
    gt_azi_2 = gt_azi_arr[:, :, 1, :] # B, T, n_avb_mics
    es_azi_1 = torch.max(es_as_1, 3)[1] # B, T, n_avb_mics
    es_azi_2 = torch.max(es_as_2, 3)[1] # B, T, n_avb_mics

    mask_1 = torch.ones((es_azi_1.shape[0], es_azi_1.shape[1], es_azi_1.shape[2]), device=es_as_1.device) # filter -1 value
    mask_2 = torch.ones((es_azi_2.shape[0], es_azi_2.shape[1], es_azi_2.shape[2]), device=es_as_1.device)  
    mask_1[gt_azi_1 == -1] = 0
    mask_2[gt_azi_2 == -1] = 0
    masked_gt_azi_1 = gt_azi_1 * mask_1 # B, T, n_avb_mics
    masked_gt_azi_2 = gt_azi_2 * mask_2
    masked_es_azi_1 = es_azi_1 * mask_1
    masked_es_azi_2 = es_azi_2 * mask_2
    abs_err_azi_1 = torch.abs(masked_gt_azi_1-masked_es_azi_1)
    abs_err_azi_2 = torch.abs(masked_gt_azi_2-masked_es_azi_2)
    num_pred_1 = mask_1.sum()
    num_pred_2 = mask_2.sum()
    mae_1 = torch.sum(abs_err_azi_1) / num_pred_1 # [1]
    mae_2 = torch.sum(abs_err_azi_2) / num_pred_2 # [1]
    num_acc_1 = torch.where(abs_err_azi_1 <= 5, 1, 0).sum() - torch.sum(mask_1 == 0)
    num_acc_2 = torch.where(abs_err_azi_2 <= 5, 1, 0).sum() - torch.sum(mask_2 == 0)

    return mae_1, mae_2, num_acc_1, num_acc_2, num_pred_1, num_pred_2

def dist_err_2_source(gt_dist_arr, es_ds_1, es_ds_2, dist_tolerance=20):
    """
    gt_dist_arr: [B, T, S]
    es_ds_1 / es_ds_2: [B, T, F] F: feature dimension of the likelihood-based coding
    """
    gt_dist_1 = gt_dist_arr[:, :, 0] # B, T
    gt_dist_2 = gt_dist_arr[:, :, 1] # B, T
    es_dist_1 = torch.max(es_ds_1, 2)[1] # B, T
    es_dist_2 = torch.max(es_ds_2, 2)[1] # B, T

    mask_1 = torch.ones((es_dist_1.shape[0], es_dist_1.shape[1]), device=es_ds_1.device) # filter -1 value
    mask_2 = torch.ones((es_dist_2.shape[0], es_dist_2.shape[1]), device=es_ds_1.device)  
    mask_1[gt_dist_1 == -1] = 0
    mask_2[gt_dist_2 == -1] = 0
    masked_gt_dist_1 = gt_dist_1 * mask_1 # B, T
    masked_gt_dist_2 = gt_dist_2 * mask_2
    masked_es_dist_1 = es_dist_1 * mask_1
    masked_es_dist_2 = es_dist_2 * mask_2
    abs_err_azi_1 = torch.abs(masked_gt_dist_1-masked_es_dist_1)
    abs_err_azi_2 = torch.abs(masked_gt_dist_2-masked_es_dist_2)
    num_pred_1 = mask_1.sum()
    num_pred_2 = mask_2.sum()
    mae_1 = torch.sum(abs_err_azi_1) / num_pred_1 # [1]
    mae_2 = torch.sum(abs_err_azi_2) / num_pred_2 # [1]
    num_acc_1 = torch.where(abs_err_azi_1 <= dist_tolerance, 1, 0).sum() - torch.sum(mask_1 == 0)
    num_acc_2 = torch.where(abs_err_azi_2 <= dist_tolerance, 1, 0).sum() - torch.sum(mask_2 == 0)

    return mae_1, mae_2, num_acc_1, num_acc_2, num_pred_1, num_pred_2


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR without PIT training.
    Args:
        source: [B, C, T] a tensor
        estimate_source: [B, C, T] a tensor
        B:batch_size C: channel T:lenght of audio 
        in this case C==1  only single channel
    """
    # assert source.size() == estimate_source.size()
    B, C, T = source.size()

    # Step 1. Zero-mean norm
    # num_samples = source_lengths.view(-1, 1, 1).float() ?# [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / T
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / T
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate


    # Step 2. SI-SNR without PIT
    # reshape to use broadcast
    s_target = zero_mean_target # [B, C, T]
    s_estimate = zero_mean_estimate # [B, C, T]
    # s_target = source
    # s_estimate = estimate_source


    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # print('s_target:',pair_wise_proj[0][0][0:10])
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]

    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]

    # print('888',pair_wise_si_snr.shape)
    si_snr = torch.mean(pair_wise_si_snr, dim=0)

    # si_snr = torch.sum(pair_wise_si_snr,dim=0)

    return round(float(si_snr.cpu().numpy()[0]),4)

def cal_si_snr_np(source, estimate_source):
    """Calculate SI-SNR without PIT training.
    Args:
        source: [B, C, T] a tensor
        estimate_source: [B, C, T] a tensor
        B:batch_size C: channel T:lenght of audio 
        in this case C==1  only single channel
    """
    # assert source.size() == estimate_source.size()
    B, C, T = source.shape

    # Step 1. Zero-mean norm
    # num_samples = source_lengths.view(-1, 1, 1).float() ?# [B, 1, 1]
    mean_target = np.sum(source, axis=2, keepdims=True) / T
    mean_estimate = np.sum(estimate_source, axis=2, keepdims=True) / T
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate


    # Step 2. SI-SNR without PIT
    # reshape to use broadcast
    s_target = zero_mean_target # [B, C, T]
    s_estimate = zero_mean_estimate # [B, C, T]
    # s_target = source
    # s_estimate = estimate_source


    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = np.sum(s_estimate * s_target, axis=2, keepdims=True)  # [B, C, 1]
    s_target_energy = np.sum(s_target ** 2, axis=2, keepdims=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # print('s_target:',pair_wise_proj[0][0][0:10])
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]

    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = np.sum(pair_wise_proj ** 2, axis=2) / (np.sum(e_noise ** 2, axis=2) + EPS)
    pair_wise_si_snr = 10 * np.log10(pair_wise_si_snr + EPS)  # [B, C]

    # print('888',pair_wise_si_snr.shape)
    si_snr = np.mean(pair_wise_si_snr, axis=0)

    # si_snr = np.sum(pair_wise_si_snr,axis=0)

    return round(float(si_snr[0]),4)

def pow_p_norm(signal):
    """Compute 2 Norm"""
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)

def pow_norm(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = torch.mean(signal, dim=-1, keepdim=True)
    signal = signal - mean
    return signal

def si_sdr(estimated, original):
    # estimated = remove_dc(estimated)
    # original = remove_dc(original)
    target = pow_norm(estimated, original) * original / (pow_p_norm(original) + EPS)
    noise = estimated - target
    sdr = 10 * torch.log10(pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
    return sdr.squeeze_(dim=-1)

# Minimize negative SI-SDR
def permute_si_sdr(est, src):
    """ Caculate SI-SDR with PIT.
    Args:
        est: [batch_size, nspk, length]
        src: [batch_size, nspk, length]
    """
    assert est.size() == src.size()
    nspk = est.size(1)
    # reshape source to [batch_size, 1, nspk, length]
    src = torch.unsqueeze(src, dim=1)
    # reshape estimation to [batch_size, nspk, 1, length]
    est = torch.unsqueeze(est, dim=2)
    pair_wise_sdr = si_sdr(est, src) # [batch_size, nspk, nspk]
    # permutation, [nspk!, nspk]
    perms = torch.tensor(list(permutations(range(nspk))), dtype=torch.long)
    index = torch.unsqueeze(perms, dim=-1)
    # one-hot, [nspk!, nspk, nspk]
    perms_one_hot = torch.zeros((*perms.size(), nspk)).scatter_(-1, index, 1)
    perms_one_hot = perms_one_hot.to(est.device)
    # einsum([batch_size, nspk, nspk], [nspk!, nspk, nspk]) -> [batch_size, nspk!]
    sdr_set = torch.einsum('bij,pij->bp', [pair_wise_sdr, perms_one_hot])
    # max_sdr_idx = torch.argmax(sdr_set, dim=-1)
    max_sdr, _ = torch.max(sdr_set, dim=-1)
    avg_loss = 0.0 - torch.mean(max_sdr / nspk)
    return avg_loss

def assignment_si_sdr(est, ref):
    """ Solve the assignment problem when computing SI-SDR.
    Args:
        est: [batch_size, est_num_sources=3, length]
        ref: [batch_size, ref_num_sources=2, length]
    """
    ref_num_sources = ref.size(1)
    est_num_sources = est.size(1)
    losses = []
    idxs = []
    # This itertools call returns all possible assignments from estimates to
    # references, E.g. itertools.product(range(2), repeat=3) produces:
    # (0, 0, 0)
    # (0, 0, 1)
    # (0, 1, 0)
    # (0, 1, 1)
    # (1, 0, 0)
    # (1, 0, 1)
    # (1, 1, 0)
    # (1, 1, 1)
    for idx in product(range(ref_num_sources), repeat=est_num_sources):
        # mix_matrix's shape: [1, ref_num_sources, est_num_sources]
        mix_matrix = torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(idx), num_classes=ref_num_sources).transpose(1,0).float(), dim=0)
        # estimate_mixed's shape: [batch, ref_num_sources, ...].
        estimate_mixed = torch.matmul(mix_matrix, est)
        # losses[0]: [batch, ref_num_sources]        
        losses.append(torch.mean(si_sdr(estimate_mixed, ref), dim=1, keepdim=True))
        idxs.append(idx)
    loss_matrix = torch.cat(losses, dim=1)
    # loss_matrix's shape: [batch, len(idxs)].
    idx_argmin = torch.argmin(loss_matrix, dim=1)
    idx_argmin = torch.unsqueeze(idx_argmin, 1)
    # idx_argmin's shape: [batch, 1].

    def th_gather_1d(x, indices):
        x_gather = torch.index_select(x, 1, indices)
        return x_gather
    def th_gather_2d(param, indices):
        # param: [batch, row_size, col_size]
        # indices: [batch, 2]
        # return: [batch, horizon]
        indices = indices[:, 1]
        return torch.stack([param[i, indices[i]] for i in range(indices.shape[0])], 0)

    print(f'loss_matrix {loss_matrix.shape}')
    print(f'loss_matrix {loss_matrix}')
    print(f'A idx_argmin {idx_argmin.shape}')
    print(f'A idx_argmin {idx_argmin}')
    loss_best_mixture = torch.gather(loss_matrix, 1, idx_argmin).squeeze() # [B]
    print(f'loss_best_mixture {loss_best_mixture.shape}')
    print(f'loss_best_mixture {loss_best_mixture}')
    # idxs_tf [len(idxs), est_num_sources]
    idxs_tf = torch.cat([torch.unsqueeze(torch.tensor(idx), dim=0) for idx in idxs], dim=0)
    idxs_tf = torch.unsqueeze(idxs_tf, 0).repeat(batch, 1, 1)
    print(f'idxs_tf {idxs_tf.shape}')
    # idxs_tf's shape: [batch, len(idxs), est_num_sources]. 
    idx_argmin = torch.stack([idx_argmin for i in range(est_num_sources)], dim=-1)
    print(f'B idx_argmin {idx_argmin.shape}')
    print(f'B idx_argmin {idx_argmin}')
    idxs_best = torch.gather(idxs_tf, 1, idx_argmin).squeeze()
    # idxs_best is shape [batch, est_num_sources].
    print(f'idxs_best {idxs_best.shape}')
    mix_matrix = torch.nn.functional.one_hot(idxs_best, num_classes=ref_num_sources).to(torch.float32)
    print(f'mix_matrix {mix_matrix.shape}')
    # mix_matrix's shape [batch, ref_num_sources, est_num_sources].

    return loss_best_mixture, mix_matrix

def cal_pesq(source, estimate_source):
    """Calculate PESQ without PIT training.
    Args:
        source: [B, T]
        estimate_source: [B, T]
    """
    # assert source.size() == estimate_source.size()

    sr = 16000
    # print(source.shape)
    batch_size, _ = source.shape
    pesq_sum = 0
    for batch_idx in range(batch_size):
        pesq_sum += pesq(sr, source[batch_idx, :], estimate_source[batch_idx, :], 'wb')

    return (pesq_sum / batch_size)

def audioread(path, segment=64000, fs = 16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        wave_data = librosa.resample(wave_data,sr,fs)
    return wave_data

def loss_energy(x, eps=1e-8, fs=16000):
    # [B, T]
    """
    Arguments:
    x: estimated signal, [B, T] tensor
    average by duration (seconds)
    """
    return torch.mean(10 * torch.log10(pow_p_norm(x) * fs / x.shape[-1] + eps))

def split_spec_into_subbands(sig, freq_bound, win_size, hop_size, fs=16000):
    """
    Args:
        sig: [ch, T]  
    """
    import apkit    
    # tf.shape: [C, num_frames, win_size] tf.dtype: complex128
    tf = apkit.stft(sig, apkit.cola_hamming, win_size, hop_size, last_sample=True)
    # trim freq bins
    
    fbin_bound = int(freq_bound * win_size / fs)            
    tf_below = tf[:,:, :fbin_bound] # 0-4kHz
    tf_above = tf[:,:, fbin_bound:] # > 4kHz
    return tf_below, tf_above

def split_spec_into_subbands_timedomain(sig, freq_bound, win_size, hop_size, fs=16000):
    """
    Args:
        sig: [ch, T]  
    """
    import apkit    
    # tf.shape: [C, num_frames, win_size] tf.dtype: complex128
    tf = apkit.stft(sig, apkit.cola_hamming, win_size, hop_size, last_sample=True)
    # trim freq bins
    fbin_bound = int(freq_bound * win_size / fs)            

    tf_below = tf[:,:, :fbin_bound] # 0-4kHz
    tf_above = tf[:,:, fbin_bound:] # > 4kHz

    tf_below = np.pad(tf_below, ((0, 0), (0, 0), (0, win_size-fbin_bound)))
    tf_above = np.pad(tf_above, ((0, 0), (0, 0), (fbin_bound, 0)))
    # tf_below.shape: [C, num_frames, win_size] tf_above.shape: [C, num_frames, win_size]
    sig_below = apkit.istft(tf_below, hop_size) # [C, T]
    sig_above = apkit.istft(tf_above, hop_size) # [C, T]

    # write_wav("/Work21/2021/fuyanjie/pycode/MIMO_DBnet/1-10new/testlt4k.wav", sig_below[0])
    # write_wav("/Work21/2021/fuyanjie/pycode/MIMO_DBnet/1-10new/testgt4k.wav", sig_above[0])
    return sig_below, sig_above

if __name__ == '__main__':
    # e1 = np.random.randn(1, 64000)
    # e2 = np.random.randn(1, 1, 32000)
    # c1 = np.random.randn(1, 1, 32000)
    # c2 = np.random.randn(1, 1, 32000)
    # print(f'e1.shape {e1.shape} e2.shape {e2.shape}')
    # print(f'c1.shape {c1.shape} c2.shape {c2.shape}')
    # print(cal_si_snr_np(e1, c1))
    # print(cal_si_snr_np(e2, c2))
    # es 71, 82 
    gt_azi_arr = torch.tensor([[[62,62]]])
    # print(f'{gt_azi_arr.shape} {es_as_1.shape} {es_as_2.shape}')
    # print(doa_err_2_source(gt_azi_arr, es_as_1, es_as_2))