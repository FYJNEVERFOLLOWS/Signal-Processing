from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn


class ComMagMse(_Loss):
    def forward(self, esti, label):
        # print("esti.shape:{}, label.shape:{}".format(esti.shape, label.shape))
        # label size:torch.Size([batch, 2, frame, freq])
        if label.ndim == 2:
            # print("label.shape:{}".format(label.shape))
            # batch_size = label.size()[0]
            fft_num = int(esti.shape[-1] // 2 *4 )
            win_size = fft_num
            win_shift = fft_num // 2 
            # print("fft_num:{}, win_size:{},win_shift:{}".format(fft_num, win_size, win_shift))
            # label = label.view(batch_size, wav_len)
            #print("inpt.shape:{}".format(inpt.shape))
            label_stft = torch.stft(label, fft_num, win_shift, win_size, torch.hann_window(win_size).cuda())
            label_stft = label_stft.permute(0, 3, 2, 1).cuda()
            label_mag, label_phase = torch.norm(label_stft, dim=1) ** 0.5, torch.atan2(label_stft[:, -1, ...], label_stft[:, 0, ...])
            label_stft = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1).cuda()
            # print("label_stft.shape:{}".format(label_stft.shape))
            label = label_stft
        mask_for_loss = []
        utt_num = esti.size()[0]
        with torch.no_grad():
            for i in range(utt_num):
                tmp_mask = torch.ones((esti.shape[2], esti.size()[-1]), dtype=esti.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
            com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
        mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
        loss1 = (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
        loss2 = (((esti - label)**2.0)*com_mask_for_loss).sum() / com_mask_for_loss.sum()
        return 0.5*(loss1 + loss2)

class LJJComMagMse(_Loss):
    def penalty(self,esti,label,alpha,mask_for_loss):
        diff = label-esti
        return ((torch.where(diff>0,alpha*diff,diff)**2.0)*mask_for_loss).sum()/mask_for_loss.sum()

    def forward(self, esti, label):
        # print("esti.shape:{}, label.shape:{}".format(esti.shape, label.shape))
        # label size:torch.Size([batch, 2, frame, freq])
        if label.ndim == 2:
            # print("label.shape:{}".format(label.shape))
            # batch_size = label.size()[0]
            fft_num = int(esti.shape[-1] // 2 *4 )
            win_size = fft_num
            win_shift = fft_num // 2 
            # print("fft_num:{}, win_size:{},win_shift:{}".format(fft_num, win_size, win_shift))
            # label = label.view(batch_size, wav_len)
            #print("inpt.shape:{}".format(inpt.shape))
            label_stft = torch.stft(label, fft_num, win_shift, win_size, torch.hann_window(win_size).cuda())
            label_stft = label_stft.permute(0, 3, 2, 1).cuda()
            label_mag, label_phase = torch.norm(label_stft, dim=1) ** 0.5, torch.atan2(label_stft[:, -1, ...], label_stft[:, 0, ...])
            label_stft = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1).cuda()
            # print("label_stft.shape:{}".format(label_stft.shape))
            label = label_stft
        mask_for_loss = []
        utt_num = esti.size()[0]
        with torch.no_grad():
            for i in range(utt_num):
                tmp_mask = torch.ones((esti.shape[2], esti.size()[-1]), dtype=esti.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
            com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
        mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
        # loss1 = (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
        # loss2 = (((esti - label)**2.0)*com_mask_for_loss).sum() / com_mask_for_loss.sum()
        loss1 = self.penalty(mag_esti,mag_label,3,mask_for_loss)
        loss2 = self.penalty(esti,label,2,com_mask_for_loss)
        return 0.5*(loss1 + loss2)

if __name__ == "__main__":
    loss_func = LJJComMagMse()