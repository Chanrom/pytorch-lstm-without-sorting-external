# pytorch-lstm-without-sorting-externally

In Pytorch, if we use ``torch.nn.LSTM`` and the input data are padded (i.e. a batch of variable-length sentences), then these data must be sorted by length and wrapped by ``torch.nn.utils.rnn.PackedSequence``. Sometimes it's inconvenient to do that if we have n-pair data and want them to be interactive.

``lstm.py`` implements a LSTM that can all these things internally. It wraps a standard Pytorch LSTM and sorts the input data automatically. The drawback is it slower than the standard Pytorch LSTM by 10%.
