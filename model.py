import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.utils.model_zoo as model_zoo

from allennlp.modules.elmo import Elmo

class Atten(nn.Module):
    def __init__(self, conf):
        super(Atten, self).__init__()
        self.conf = conf
        self.w = nn.Conv1d(self.conf.cnn_dim, self.conf.cnn_dim, 1)
        self.temper = np.power(self.conf.cnn_dim, 0.5)
        self.dropout = nn.Dropout(self.conf.attn_dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, q, v):
        q_ = self.w(q.transpose(1, 2)).transpose(1, 2)
        attn = torch.bmm(q_, v.transpose(1, 2)) / self.temper
        vr = torch.bmm(self.dropout(self.softmax(attn)), v)
        qr = torch.bmm(self.dropout(self.softmax(attn.transpose(1, 2))), q)
        vr = torch.topk(vr, k=self.conf.attn_topk, dim=1)[0]
        vr = vr.view(vr.size(0), -1)
        qr = torch.topk(qr, k=self.conf.attn_topk, dim=1)[0]
        qr = qr.view(qr.size(0), -1)
        return qr, vr, attn

class Highway(nn.Module):
    def __init__(self, size):
        super(Highway, self).__init__()
        self.highway_linear = nn.Linear(size, size)
        self.gate_linear = nn.Linear(size, size)
        self.nonlinear = nn.ReLU()

    def forward(self, input):
        gate = F.sigmoid(self.gate_linear(input))
        m = self.nonlinear(self.highway_linear(input))
        return gate * m + (1 - gate) * input

class RNNLayer(nn.Module):
    def __init__(self, in_dim, conf):
        super(RNNLayer, self).__init__()
        self.conf = conf
        self.rnn = nn.GRU(in_dim, in_dim, num_layers=1,
                            dropout=self.conf.cnn_dropout, bidirectional=True)
        self.conv = nn.Conv1d(in_dim*2, in_dim, 1)
        nn.init.xavier_uniform(self.conv.weight)
        self.conv.bias.data.fill_(0)
        self.dropout = nn.Dropout(self.conf.cnn_dropout)

    def forward(self, input, length, hidden=None):
        lens, indices = torch.sort(length, 0, True)
        maxlen = lens[0]
        outputs, hidden_t = self.rnn(pack(input[indices], lens.tolist(), batch_first=True), hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices]
        size = outputs.size()
        outputs = F.pad(outputs.unsqueeze(0), (0,0,0,self.conf.max_sent_len-maxlen)).view(size[0],-1,size[2])
        outputs = self.conv(self.dropout(outputs.transpose(1, 2))).transpose(1, 2)
        return outputs + input

class CNNLayer(nn.Module):
    def __init__(self, conf, in_dim, k, res=True):
        super(CNNLayer, self).__init__()
        self.conf = conf
        self.res = res
        self.conv = nn.Conv1d(in_dim, in_dim*2, k, stride=1, padding=k//2)
        nn.init.xavier_uniform(self.conv.weight)
        self.conv.bias.data.fill_(0)
        self.dropout = nn.Dropout(self.conf.cnn_dropout)

    def forward(self, input):
        output = self.dropout(input.transpose(1, 2))
        tmp = self.conv(output)
        if tmp.size(2) > output.size(2):
            output = tmp[:, :, 1:]
        else:
            output = tmp
        output = output.transpose(1, 2)
        a, b = torch.chunk(output, 2, dim=2)
        output = a * nn.functional.sigmoid(b)
        if self.res:
            output = output + input
        return output

class CharLayer(nn.Module):
    def __init__(self, char_table, conf):
        super(CharLayer, self).__init__()
        self.conf = conf
        lookup, length = char_table
        self.char_embed = nn.Embedding(self.conf.char_num, self.conf.char_embed_dim, padding_idx=self.conf.char_padding_idx)
        self.lookup = nn.Embedding(lookup.size(0), lookup.size(1))
        self.lookup.weight.data.copy_(lookup)
        self.lookup.weight.requires_grad = False
        self.convs = nn.ModuleList()
        for i in range(self.conf.char_filter_num):
            self.convs.append(nn.Conv1d(
                self.conf.char_embed_dim, self.conf.char_enc_dim, self.conf.char_filter_dim[i],
                stride=1, padding=self.conf.char_filter_dim[i]//2
            ))
            nn.init.xavier_uniform(self.convs[i].weight)
        self.nonlinear = nn.Tanh()
        self.mask = nn.Embedding(lookup.size(0), self.conf.char_hid_dim)
        self.mask.weight.data.fill_(1)
        self.mask.weight.data[0].fill_(0)
        self.mask.weight.data[1].fill_(0)
        self.mask.weight.requires_grad = False
        self.highway = Highway(self.conf.char_hid_dim)
        del lookup
        del length

    def forward(self, input):
        charseq = self.lookup(input).long().view(input.size(0)*input.size(1), -1)
        charseq = self.char_embed(charseq).transpose(1, 2)
        conv_out = []
        for i in range(self.conf.char_filter_num):
            tmp = self.nonlinear(self.convs[i](charseq))
            if tmp.size(2) > charseq.size(2):
                tmp = tmp[:, :, 1:]
            tmp = torch.topk(tmp, k=1)[0]
            conv_out.append(torch.squeeze(tmp, dim=2))
        hid = torch.cat(conv_out, dim=1)
        hid = self.highway(hid)
        hid = hid.view(input.size(0), input.size(1), -1)
        mask = self.mask(input)
        hid = hid * mask
        return hid

class ElmoLayer(nn.Module):
    def __init__(self, char_table, conf):
        super(ElmoLayer, self).__init__()
        self.conf = conf
        lookup, length = char_table
        self.lookup = nn.Embedding(lookup.size(0), lookup.size(1))
        self.lookup.weight.data.copy_(lookup)
        self.lookup.weight.requires_grad = False
        self.elmo = Elmo(
            os.path.expanduser(self.conf.elmo_options), os.path.expanduser(self.conf.elmo_weights),
            num_output_representations=2, do_layer_norm=False, dropout=self.conf.embed_dropout
        )
        for p in self.elmo.parameters():
            p.requires_grad = False
        self.w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.gamma = nn.Parameter(torch.ones(1))
        self.conv = nn.Conv1d(1024, self.conf.elmo_dim, 1)
        nn.init.xavier_uniform(self.conv.weight)
        self.conv.bias.data.fill_(0)

    def forward(self, input):
        charseq = self.lookup(input).long()
        res = self.elmo(charseq)['elmo_representations']
        w = F.softmax(self.w, dim=0)
        res = self.gamma * (w[0] * res[0] + w[1] * res[1])
        res = self.conv(res.transpose(1, 2)).transpose(1, 2)
        return res

class ArgEncoder(nn.Module):
    def __init__(self, conf, we_tensor, char_table=None, sub_table=None, use_cuda=False, attnvis=False):
        super(ArgEncoder, self).__init__()
        self.conf = conf
        self.attnvis = attnvis
        if self.conf.use_rnn:
            self.usecuda = use_cuda
        self.embed = nn.Embedding(we_tensor.size(0), we_tensor.size(1))
        self.embed.weight.data.copy_(we_tensor)
        self.embed.weight.requires_grad = False
        if self.conf.need_char:
            self.charenc = CharLayer(char_table, self.conf)
        if self.conf.need_sub:
            self.charenc = CharLayer(sub_table, self.conf)
        if self.conf.need_elmo:
            self.elmo = ElmoLayer(char_table, self.conf)
        self.dropout = nn.Dropout(self.conf.embed_dropout)

        self.block1 = nn.ModuleList()
        self.block2 = nn.ModuleList()
        self.attn = Atten(self.conf)
        for i in range(self.conf.cnn_layer_num):
            if self.conf.use_rnn:
                self.block1.append(RNNLayer(self.conf.cnn_dim, self.conf))
                self.block2.append(RNNLayer(self.conf.cnn_dim, self.conf))
            else:
                self.block1.append(CNNLayer(self.conf, self.conf.cnn_dim, self.conf.cnn_kernal_size[i]))
                self.block2.append(CNNLayer(self.conf, self.conf.cnn_dim, self.conf.cnn_kernal_size[i]))
    
    def forward(self, a1, a2):
        if self.conf.use_rnn:
            len1 = torch.LongTensor([torch.max(a1[i,:].data.nonzero())+1 for i in range(a1.size(0))])
            len2 = torch.LongTensor([torch.max(a2[i,:].data.nonzero())+1 for i in range(a2.size(0))])
            if self.usecuda:
                len1 = len1.cuda()
                len2 = len2.cuda()
        arg1repr = self.embed(a1)
        arg2repr = self.embed(a2)
        if self.conf.need_char or self.conf.need_sub:
            char1 = self.charenc(a1)
            char2 = self.charenc(a2)
            arg1repr = torch.cat((arg1repr, char1), dim=2)
            arg2repr = torch.cat((arg2repr, char2), dim=2)
        if self.conf.need_elmo:
            arg1repr = torch.cat((arg1repr, self.elmo(a1)), dim=2)
            arg2repr = torch.cat((arg2repr, self.elmo(a2)), dim=2)
        arg1repr = self.dropout(arg1repr)
        arg2repr = self.dropout(arg2repr)
        outputs = []
        attns = []
        for i in range(self.conf.cnn_layer_num):
            if self.conf.use_rnn:
                arg1repr = self.block1[i](arg1repr, len1)
                arg2repr = self.block2[i](arg2repr, len2)
            else:
                arg1repr = self.block1[i](arg1repr)
                arg2repr = self.block2[i](arg2repr)
            outputc1, outputc2, attnw = self.attn(arg1repr, arg2repr)
            outputs.append(outputc1)
            outputs.append(outputc2)
            attns.append(attnw)
        if self.attnvis:
            return torch.cat(outputs, 1), attns
        else:
            return torch.cat(outputs, 1)

class Classifier(nn.Module):
    def __init__(self, nclass, conf):
        super(Classifier, self).__init__()
        self.conf = conf
        self.dropout = nn.Dropout(self.conf.clf_dropout)
        self.fc = nn.ModuleList()
        if self.conf.clf_fc_num > 0:
            self.fc.append(nn.Linear(self.conf.pair_rep_dim, self.conf.clf_fc_dim))
            for i in range(self.conf.clf_fc_num - 1):
                self.fc.append(nn.Linear(self.conf.clf_fc_dim, self.conf.clf_fc_dim))
            lastfcdim = self.conf.clf_fc_dim
        else:
            lastfcdim = self.conf.pair_rep_dim
        self.lastfc = nn.Linear(lastfcdim, nclass)
        self.nonlinear = nn.Tanh()
        self._init_weight()

    def _init_weight(self):
        for i in range(self.conf.clf_fc_num):
            self.fc[i].bias.data.fill_(0)
            nn.init.uniform(self.fc[i].weight, -0.01, 0.01)
        self.lastfc.bias.data.fill_(0)
        nn.init.uniform(self.lastfc.weight, -0.01, 0.01)

    def forward(self, input):
        output = input
        for i in range(self.conf.clf_fc_num):
            output = self.nonlinear(self.dropout(self.fc[i](output)))
        output = self.lastfc(self.dropout(output))
        return output

class IDRCModel(nn.Module):
    def __init__(self, conf, we_tensor, char_table, sub_table, use_cuda):
        super(IDRCModel, self).__init__()
        self.enc = ArgEncoder(conf, we_tensor, char_table, sub_table, use_cuda)
        self.clf = Classifier(conf.clf_class_num, conf)

    def forward(self, arg1, arg2):
        return self.clf(self.enc(arg1, arg2))

def test():
    from data import Data
    from config import Config
    conf = Config()
    usecuda = True
    we = torch.load('./data/processed/ji/we.pkl')
    char_table = None
    sub_table = None
    if conf.need_char or conf.need_elmo:
        char_table = torch.load('./data/processed/ji/char_table.pkl')
    if conf.need_sub:
        sub_table = torch.load('./data/processed/ji/sub_table.pkl')
    model = IDRCModel(conf, we, char_table, sub_table, usecuda)
    if usecuda:
        model.cuda()
    d = Data(usecuda, conf)
    for a1, a2, sense, conn in d.train_loader:
        if usecuda:
            a1, a2 = a1.cuda(), a2.cuda()
        a1, a2 = Variable(a1), Variable(a2)
        break
    model.eval()
    out = model(a1, a2)
    print(out)

if __name__ == '__main__':
    test()