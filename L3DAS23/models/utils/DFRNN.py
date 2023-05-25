import torch
from torch import nn
from torch.nn.functional import fold, unfold
import numpy as np

from asteroid import complex_nn
from asteroid.utils import has_arg
from asteroid.masknn import activations, norms
from asteroid.masknn._dccrn_architectures import DCCRN_ARCHITECTURES
from asteroid.masknn.base import BaseDCUMaskNet
from asteroid.masknn.norms import CumLN, GlobLN


class SingleRNN(nn.Module):
    """Module for a RNN block.

    Inspired from https://github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False
    ):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inp):
        """Input shape [batch, seq, feats]"""
        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output


class MulCatRNN(nn.Module):
    """MulCat RNN block from [1].

    Composed of two RNNs, returns ``cat([RNN_1(x) * RNN_2(x), x])``.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    References
        [1] Eliya Nachmani, Yossi Adi, & Lior Wolf. (2020). Voice Separation with an Unknown Number of Multiple Speakers.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False
    ):
        super(MulCatRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn1 = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )
        self.rnn2 = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1) + self.input_size

    def forward(self, inp):
        """Input shape [batch, seq, feats]"""
        self.rnn1.flatten_parameters()  # Enables faster multi-GPU training.
        self.rnn2.flatten_parameters()  # Enables faster multi-GPU training.
        rnn_output1, _ = self.rnn1(inp)
        rnn_output2, _ = self.rnn2(inp)
        return torch.cat((rnn_output1 * rnn_output2, inp), 2)


class StackedResidualRNN(nn.Module):
    """Stacked RNN with builtin residual connection.
    Only supports forward RNNs.
    See StackedResidualBiRNN for bidirectional ones.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        n_units (int): Number of units in recurrent layers. This will also be
            the expected input size.
        n_layers (int): Number of recurrent layers.
        dropout (float): Dropout value, between 0. and 1. (Default: 0.)
        bidirectional (bool): If True, use bidirectional RNN, else
            unidirectional. (Default: False)
    """

    def __init__(self, rnn_type, n_units, n_layers=4, dropout=0.0, bidirectional=False):
        super(StackedResidualRNN, self).__init__()
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        assert bidirectional is False, "Bidirectional not supported yet"
        self.bidirectional = bidirectional

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                SingleRNN(
                    rnn_type, input_size=n_units, hidden_size=n_units, bidirectional=bidirectional
                )
            )
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """Builtin residual connections + dropout applied before residual.
        Input shape : [batch, time_axis, feat_axis]
        """
        for rnn in self.layers:
            rnn_out = rnn(x)
            dropped_out = self.dropout_layer(rnn_out)
            x = x + dropped_out
        return x


class StackedResidualBiRNN(nn.Module):
    """Stacked Bidirectional RNN with builtin residual connection.
    Residual connections are applied on both RNN directions.
    Only supports bidiriectional RNNs.
    See StackedResidualRNN for unidirectional ones.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        n_units (int): Number of units in recurrent layers. This will also be
            the expected input size.
        n_layers (int): Number of recurrent layers.
        dropout (float): Dropout value, between 0. and 1. (Default: 0.)
        bidirectional (bool): If True, use bidirectional RNN, else
            unidirectional. (Default: False)
    """

    def __init__(self, rnn_type, n_units, n_layers=4, dropout=0.0, bidirectional=True):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        assert bidirectional is True, "Only bidirectional not supported yet"
        self.bidirectional = bidirectional

        # The first layer has as many units as input size
        self.first_layer = SingleRNN(
            rnn_type, input_size=n_units, hidden_size=n_units, bidirectional=bidirectional
        )
        # As the first layer outputs 2*n_units, the following layers need
        # 2*n_units as input size
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            input_size = 2 * n_units
            self.layers.append(
                SingleRNN(
                    rnn_type,
                    input_size=input_size,
                    hidden_size=n_units,
                    bidirectional=bidirectional,
                )
            )
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """Builtin residual connections + dropout applied before residual.
        Input shape : [batch, time_axis, feat_axis]
        """
        # First layer
        rnn_out = self.first_layer(x)
        dropped_out = self.dropout_layer(rnn_out)
        x = torch.cat([x, x], dim=-1) + dropped_out
        # Rest of the layers
        for rnn in self.layers:
            rnn_out = rnn(x)
            dropped_out = self.dropout_layer(rnn_out)
            x = x + dropped_out
        return x


class DPRNNBlock(nn.Module):
    """Dual-Path RNN Block as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_size (int): Number of hidden neurons in the RNNs.
        norm_type (str, optional): Type of normalization to use. To choose from
            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN.
        rnn_type (str, optional): Type of RNN used. Choose from ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers used in each RNN.
        dropout (float, optional): Dropout ratio. Must be in [0, 1].

    References
        [1] "Dual-path RNN: efficient long sequence modeling for
        time-domain single-channel speech separation", Yi Luo, Zhuo Chen
        and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
    """

    def __init__(
        self,
        intra_chan,
        inter_chan,
        hid_size,
        norm_type="gLN",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
    ):
        super(DPRNNBlock, self).__init__()
        self.intra_RNN = SingleRNN(
            rnn_type,
            intra_chan,
            hid_size,
            num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.inter_RNN = SingleRNN(
            rnn_type,
            inter_chan,
            hid_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.intra_linear = nn.Linear(self.intra_RNN.output_size, intra_chan)
        self.intra_norm = norms.get(norm_type)(intra_chan)

        self.inter_linear = nn.Linear(self.inter_RNN.output_size, inter_chan)
        self.inter_norm = norms.get(norm_type)(inter_chan)

    def forward(self, x):
        """Input shape : [batch, feats, chunk_size, num_chunks]"""
        # batch*mics, channels, seq_len, freq_len
        #B, N, K, L = x.size()
        BN, C, S, F = x.size()
        output = x  # for skip connection
        # Intra-chunk processing,Freq
        x = x.reshape(BN*C, S, F)
        x = self.intra_RNN(x)
        x = self.intra_linear(x)
        x = x.reshape(BN, C, S, F).transpose(1, -1).contiguous()
        x = self.intra_norm(x)
        x = x.transpose(1, -1)
        output = output + x
        # Inter-chunk processing, channel
        x = output.transpose(1, -1).reshape(BN * F, S, C)
        x = self.inter_RNN(x)
        x = self.inter_linear(x)
        x = x.reshape(BN, F, S, C).transpose(1, -1).contiguous()
        x = self.inter_norm(x)
        return output + x


