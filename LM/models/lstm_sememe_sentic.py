import torch.nn as nn
from torch.nn import init
import torch
from torch.autograd import Variable
class LSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        #self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fh, self.W_c]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, emb_s, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(emb_s)
        f, i, o, o_c = torch.split(iou, iou.size(1) // 4, dim=1)
        f, i, o, o_c = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(o_c)
        c_telta = self.fx(inputs) + self.fh(child_h)
        c_telta = torch.tanh(c_telta)
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, c_telta) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c)) + torch.mul(o_c, torch.tanh(self.W_c(emb_s)))
        return (c, h)

    def forward(self, inputs, emb_s, hidden):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = hidden
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], emb_s[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack(output, 0), hx

class SememeSumLstm(nn.Module):
    def __init__(self, sememe_dim, mem_dim):
        super(SememeSumLstm, self).__init__()
        self.in_dim = sememe_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.reset_parameters()
    def node_forward(self, inputs):
        iou = self.ioux(inputs)# three Wx+b
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = torch.mul(i, u)
        h = torch.mul(o, torch.tanh(c))
        return c, h
    def forward(self, inputs):
        max_time, batch_size, _ = inputs.size()
        c = []
        h = []
        for time in range(max_time):
            new_c, new_h = self.node_forward(inputs[time])
            c.append(new_c)
            h.append(new_h)
        return torch.stack(c, 0), torch.stack(h, 0)

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
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers-1, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers-1, nonlinearity=nonlinearity, dropout=dropout)
        self.LSTM = LSTM(ninp, nhid)
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
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                (weight.new_zeros(self.nlayers-1, bsz, self.nhid),
                  weight.new_zeros(self.nlayers-1, bsz, self.nhid))
                )
