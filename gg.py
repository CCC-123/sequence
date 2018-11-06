import torch
import torch.nn as nn

bilstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True)
input = torch.randn(5, 3, 10)
h0 = torch.randn(4, 3, 20)
c0 = torch.randn(4, 3, 20)
output, (hn, cn) = bilstm(input, (h0, c0))
print output