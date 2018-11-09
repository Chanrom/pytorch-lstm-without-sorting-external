import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0, bidirectional=False, rnn_type=nn.LSTM):
        super(LSTM, self).__init__()

        self.dropout = dropout
        self.rnn = rnn_type(input_size,
                            hidden_size,
                            dropout=dropout,
                            batch_first=True,
                            num_layers=num_layers,
                            bidirectional=bidirectional)


    def forward(self, x, lengths):
        """Encode padded sequences.
        Args:
            x: batch * len * hdim
            lengths: list of all sentence length
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        lengths_ts = torch.LongTensor(lengths)
        if x.is_cuda:
            lengths_ts = lengths_ts.cuda()
        _, idx_sort = torch.sort(lengths_ts, 
                                 dim=0, 
                                 descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Pack it up
        rnn_input = pack(x, 
                         list(lengths_ts[idx_sort.data]),
                         batch_first=True)

        outputs, hidden_t = self.rnn(rnn_input)

        # Unpack
        outputs, _ = unpack(outputs, batch_first=True)

        # Unsort
        outputs = outputs.index_select(0, idx_unsort)

        return outputs, hidden_t

