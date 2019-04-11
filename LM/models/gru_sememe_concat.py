import torch.nn as nn
from torch.nn import init
import torch
from torch.autograd import Variable

class GRU_sememe(nn.Module):
    def __init__(self, ninp, nhid):
        super(GRU_sememe, self).__init__()
        self.in_dim = ninp
        self.mem_dim = nhid

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(2 * self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)

        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.Uh_s]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, sememe_h, hx):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []

        for time in range(max_time):
            new_input = torch.cat([inputs[time], sememe_h[time]], dim = 1)
            next_hx = self.node_forward(new_input, hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack(output, 0), hx

class SememeSumLstm(nn.Module):
    def __init__(self, sememe_dim, mem_dim):
        super(SememeSumLstm, self).__init__()
        self.in_dim = sememe_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.reset_parameters()
    def node_forward(self, inputs):
        iou = self.ioux(inputs)# three Wx+b
        i, o = torch.split(iou, iou.size(1) // 2, dim=1)
        i, o = torch.sigmoid(i), torch.tanh(o)

        h = torch.mul(i,o)
        return h
    def forward(self, inputs):
        max_time, batch_size, _ = inputs.size()
        h = []
        for time in range(max_time):
            new_h = self.node_forward(inputs[time])
            h.append(new_h)
        return torch.stack(h, 0)

    def reset_parameters(self):
        layers = [self.ioux]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)



class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, sememe_dim, sememe_size, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.sememe_dim = sememe_dim
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.emb_sememe = nn.Embedding(sememe_size, sememe_dim)
        self.sememesumlstm = SememeSumLstm(sememe_dim, nhid)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, 'GRU')(ninp, nhid, nlayers-1, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers-1, nonlinearity=nonlinearity, dropout=dropout)
        self.LSTM = GRU_sememe(ninp, nhid)
        self.decoder = nn.Linear(nhid, ntoken)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, input_s, hidden):
        #sememe propagate
        emb = self.drop(self.encoder(input))
        new_input = None
        emb_sememe = self.drop(self.emb_sememe.weight)
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        new_input, hidden_lstm = self.LSTM(emb, input_sememe, hidden[0])
        new_input = self.drop(new_input)
        output, hidden_rnn = self.rnn(new_input, hidden[1])
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        hidden = (hidden_lstm, hidden_rnn)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.nhid),
                weight.new_zeros(self.nlayers-1, bsz, self.nhid))
